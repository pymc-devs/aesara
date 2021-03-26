import logging
import os
import shlex
import sys
import warnings
from configparser import (
    ConfigParser,
    InterpolationError,
    NoOptionError,
    NoSectionError,
    RawConfigParser,
)
from functools import wraps
from io import StringIO
from typing import Callable, Dict, Optional, Sequence, Union

from aesara.utils import deprecated, hash_from_code


_logger = logging.getLogger("aesara.configparser")


class AesaraConfigWarning(Warning):
    def warn(cls, message, stacklevel=0):
        warnings.warn(message, cls, stacklevel=stacklevel + 3)

    warn = classmethod(warn)


class ConfigAccessViolation(AttributeError):
    """ Raised when a config setting is accessed through the wrong config instance. """


class _ChangeFlagsDecorator:
    def __init__(self, *args, _root=None, **kwargs):
        # the old API supported passing a dict as the first argument:
        if args:
            assert len(args) == 1 and isinstance(args[0], dict)
            kwargs = dict(**args[0], **kwargs)
        self.confs = {k: _root._config_var_dict[k] for k in kwargs}
        self.new_vals = kwargs
        self._root = _root

    def __call__(self, f):
        @wraps(f)
        def res(*args, **kwargs):
            with self:
                return f(*args, **kwargs)

        return res

    def __enter__(self):
        self.old_vals = {}
        for k, v in self.confs.items():
            self.old_vals[k] = v.__get__(self._root, self._root.__class__)
        try:
            for k, v in self.confs.items():
                v.__set__(self._root, self.new_vals[k])
        except Exception:
            _logger.error(f"Failed to change flags for {self.confs}.")
            self.__exit__()
            raise

    def __exit__(self, *args):
        for k, v in self.confs.items():
            v.__set__(self._root, self.old_vals[k])


class _SectionRedirect:
    """Functions as a mock property on the AesaraConfigParser.

    It redirects attribute access (to config subsectinos) to the
    new config variable properties that use "__" in their name.
    """

    def __init__(self, root, section_name):
        self._root = root
        self._section_name = section_name
        super().__init__()

    def __getattr__(self, attr):
        warnings.warn(
            f"Accessing section '{attr}' through old .-based API. "
            f"This will be removed. Use 'config.{self._section_name}__{attr}' instead.",
            DeprecationWarning,
        )
        return getattr(self._root, f"{self._section_name}__{attr}")


class AesaraConfigParser:
    """ Object that holds configuration settings. """

    def __init__(self, flags_dict: dict, aesara_cfg, aesara_raw_cfg):
        self._flags_dict = flags_dict
        self._aesara_cfg = aesara_cfg
        self._aesara_raw_cfg = aesara_raw_cfg
        self._config_var_dict: Dict = {}
        super().__init__()

    def __str__(self, print_doc=True):
        sio = StringIO()
        self.config_print(buf=sio, print_doc=print_doc)
        return sio.getvalue()

    def config_print(self, buf, print_doc=True):
        for cv in self._config_var_dict.values():
            print(cv, file=buf)
            if print_doc:
                print("    Doc: ", cv.doc, file=buf)
            print("    Value: ", cv.__get__(self, self.__class__), file=buf)
            print("", file=buf)

    def get_config_hash(self):
        """
        Return a string sha256 of the current config options. In the past,
        it was md5.

        The string should be such that we can safely assume that two different
        config setups will lead to two different strings.

        We only take into account config options for which `in_c_key` is True.
        """
        all_opts = sorted(
            [c for c in self._config_var_dict.values() if c.in_c_key],
            key=lambda cv: cv.name,
        )
        return hash_from_code(
            "\n".join(
                [
                    "{} = {}".format(cv.name, cv.__get__(self, self.__class__))
                    for cv in all_opts
                ]
            )
        )

    def add(self, name, doc, configparam, in_c_key=True):
        """Add a new variable to AesaraConfigParser.

        This method performs some of the work of initializing `ConfigParam` instances.

        Parameters
        ----------
        name: string
            The full name for this configuration variable. Takes the form
            ``"[section0__[section1__[etc]]]_option"``.
        doc: string
            A string that provides documentation for the config variable.
        configparam: ConfigParam
            An object for getting and setting this configuration parameter
        in_c_key: boolean
            If ``True``, then whenever this config option changes, the key
            associated to compiled C modules also changes, i.e. it may trigger a
            compilation of these modules (this compilation will only be partial if it
            turns out that the generated C code is unchanged). Set this option to False
            only if you are confident this option should not affect C code compilation.

        """
        if "." in name:
            raise ValueError(
                f"Dot-based sections were removed. Use double underscores! ({name})"
            )
        # Can't use hasattr here, because it returns False upon AttributeErrors
        if name in dir(self):
            raise AttributeError(
                f"A config parameter with the name '{name}' was already registered on another config instance."
            )
        configparam.doc = doc
        configparam.name = name
        configparam.in_c_key = in_c_key

        # Register it on this instance before the code below already starts accessing it
        self._config_var_dict[name] = configparam

        # Trigger a read of the value from config files and env vars
        # This allow to filter wrong value from the user.
        if not callable(configparam.default):
            configparam.__get__(self, type(self), delete_key=True)
        else:
            # We do not want to evaluate now the default value
            # when it is a callable.
            try:
                self.fetch_val_for_key(name)
                # The user provided a value, filter it now.
                configparam.__get__(self, type(self), delete_key=True)
            except KeyError:
                # This is raised because the underlying `ConfigParser` in
                # `self._aesara_cfg` does not contain an entry for the given
                # section and/or value.
                _logger.info(
                    f"Suppressed KeyError in AesaraConfigParser.add for parameter '{name}'!"
                )

        # the ConfigParam implements __get__/__set__, enabling us to create a property:
        setattr(self.__class__, name, configparam)

        # The old API used dots for accessing a hierarchy of sections.
        # The following code adds redirects that spill DeprecationWarnings
        # while allowing backwards-compatible access to dot-based subsections.
        # Because the subsectioning is recursive, redirects must be added for
        # all levels. For example: ".test", ".test.subsection".
        sections = name.split("__")
        for s in range(1, len(sections)):
            section_name = "__".join(sections[:s])
            if not hasattr(self, section_name):
                redirect = _SectionRedirect(self, section_name)
                setattr(self.__class__, section_name, redirect)

    def fetch_val_for_key(self, key, delete_key=False):
        """Return the overriding config value for a key.
        A successful search returns a string value.
        An unsuccessful search raises a KeyError

        The (decreasing) priority order is:
        - AESARA_FLAGS
        - ~./aesararc

        """

        # first try to find it in the FLAGS
        if key in self._flags_dict:
            if delete_key:
                return self._flags_dict.pop(key)
            return self._flags_dict[key]

        # next try to find it in the config file

        # config file keys can be of form option, or section__option
        key_tokens = key.rsplit("__", 1)
        if len(key_tokens) > 2:
            raise KeyError(key)

        if len(key_tokens) == 2:
            section, option = key_tokens
        else:
            section, option = "global", key
        try:
            try:
                return self._aesara_cfg.get(section, option)
            except InterpolationError:
                return self._aesara_raw_cfg.get(section, option)
        except (NoOptionError, NoSectionError):
            raise KeyError(key)

    def change_flags(self, *args, **kwargs) -> _ChangeFlagsDecorator:
        """
        Use this as a decorator or context manager to change the value of
        Aesara config variables.

        Useful during tests.
        """
        return _ChangeFlagsDecorator(*args, _root=self, **kwargs)

    def warn_unused_flags(self):
        for key in self._flags_dict.keys():
            warnings.warn(f"Aesara does not recognise this flag: {key}")


class ConfigParam:
    """Base class of all kinds of configuration parameters.

    A ConfigParam has not only default values and configurable mutability, but
    also documentation text, as well as filtering and validation routines
    that can be context-dependent.

    This class implements __get__ and __set__ methods to eventually become
    a property on an instance of AesaraConfigParser.
    """

    def __init__(
        self,
        default: Union[object, Callable[[object], object]],
        *,
        apply: Optional[Callable[[object], object]] = None,
        validate: Optional[Callable[[object], bool]] = None,
        mutable: bool = True,
    ):
        """
        Represents a configuration parameter and its associated casting and validation logic.

        Parameters
        ----------
        default : object or callable
            A default value, or function that returns a default value for this parameter.
        apply : callable, optional
            Callable that applies a modification to an input value during assignment.
            Typical use cases: type casting or expansion of '~' to user home directory.
        validate : callable, optional
            A callable that validates the parameter value during assignment.
            It may raise an (informative!) exception itself, or simply return True/False.
            For example to check the availability of a path, device or to restrict a float into a range.
        mutable : bool
            If mutable is False, the value of this config settings can not be changed at runtime.
        """
        self._default = default
        self._apply = apply
        self._validate = validate
        self._mutable = mutable
        self.is_default = True
        # set by AesaraConfigParser.add:
        self.name = None
        self.doc = None
        self.in_c_key = None

        # Note that we do not call `self.filter` on the default value: this
        # will be done automatically in AesaraConfigParser.add, potentially with a
        # more appropriate user-provided default value.
        # Calling `filter` here may actually be harmful if the default value is
        # invalid and causes a crash or has unwanted side effects.
        super().__init__()

    @property
    def default(self):
        return self._default

    @property
    def mutable(self) -> bool:
        return self._mutable

    def apply(self, value):
        """Applies modifications to a parameter value during assignment.

        Typical use cases are casting or the subsitution of '~' with the user home directory.
        """
        if callable(self._apply):
            return self._apply(value)
        return value

    def validate(self, value) -> Optional[bool]:
        """Validates that a parameter values falls into a supported set or range.

        Raises
        ------
        ValueError
            when the validation turns out negative
        """
        if not callable(self._validate):
            return True
        if self._validate(value) is False:
            raise ValueError(
                f"Invalid value ({value}) for configuration variable '{self.name}'."
            )
        return True

    def __get__(self, cls, type_, delete_key=False):
        if cls is None:
            return self
        if self.name not in cls._config_var_dict:
            raise ConfigAccessViolation(
                f"The config parameter '{self.name}' was registered on a different instance of the AesaraConfigParser."
                f" It is not accessible through the instance with id '{id(cls)}' because of safeguarding."
            )
        if not hasattr(self, "val"):
            try:
                val_str = cls.fetch_val_for_key(self.name, delete_key=delete_key)
                self.is_default = False
            except KeyError:
                if callable(self.default):
                    val_str = self.default()
                else:
                    val_str = self.default
            self.__set__(cls, val_str)
        return self.val

    def __set__(self, cls, val):
        if not self.mutable and hasattr(self, "val"):
            raise Exception(
                f"Can't change the value of {self.name} config parameter after initialization!"
            )
        applied = self.apply(val)
        self.validate(applied)
        self.val = applied


class EnumStr(ConfigParam):
    def __init__(
        self, default: str, options: Sequence[str], validate=None, mutable=True
    ):
        """Creates a str-based parameter that takes a predefined set of options.

        Parameters
        ----------
        default : str
            The default setting.
        options : sequence
            Further str values that the parameter may take.
            May, but does not need to include the default.
        validate : callable
            See `ConfigParam`.
        mutable : callable
            See `ConfigParam`.
        """
        self.all = {default, *options}

        # All options should be strings
        for val in self.all:
            if not isinstance(val, str):
                raise ValueError(f"Non-str value '{val}' for an EnumStr parameter.")
        super().__init__(default, apply=self._apply, validate=validate, mutable=mutable)

    def _apply(self, val):
        if val in self.all:
            return val
        else:
            raise ValueError(
                f"Invalid value ('{val}') for configuration variable '{self.name}'. "
                f"Valid options are {self.all}"
            )

    def __str__(self):
        return f"{self.name} ({self.all}) "


class TypedParam(ConfigParam):
    def __str__(self):
        # The "_apply" callable is the type itself.
        return f"{self.name} ({self._apply}) "


class StrParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=str, validate=validate, mutable=mutable)


class IntParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=int, validate=validate, mutable=mutable)


class FloatParam(TypedParam):
    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=float, validate=validate, mutable=mutable)


class BoolParam(TypedParam):
    """A boolean parameter that may be initialized from any of the following:
    False, 0, "false", "False", "0"
    True, 1, "true", "True", "1"
    """

    def __init__(self, default, validate=None, mutable=True):
        super().__init__(default, apply=self._apply, validate=validate, mutable=mutable)

    def _apply(self, value):
        if value in {False, 0, "false", "False", "0"}:
            return False
        elif value in {True, 1, "true", "True", "1"}:
            return True
        raise ValueError(
            f"Invalid value ({value}) for configuration variable '{self.name}'."
        )


class DeviceParam(ConfigParam):
    def __init__(self, default, *options, **kwargs):
        super().__init__(
            default, apply=self._apply, mutable=kwargs.get("mutable", True)
        )

    def _apply(self, val):
        if val == self.default or val.startswith("opencl") or val.startswith("cuda"):
            return val
        elif val.startswith("gpu"):
            raise ValueError(
                "You are tring to use the old GPU back-end. "
                "It was removed from Aesara. Use device=cuda* now. "
                "See https://github.com/pymc-devs/aesara/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29 "
                "for more information."
            )
        else:
            raise ValueError(
                'Invalid value ("{val}") for configuration '
                'variable "{self.name}". Valid options start with '
                'one of "cpu", "opencl" or "cuda".'
            )

    def __str__(self):
        return f"{self.name} ({self.default}, opencl*, cuda*) "


class ContextsParam(ConfigParam):
    def __init__(self):
        super().__init__("", apply=self._apply, mutable=False)

    def _apply(self, val):
        if val == "":
            return val
        for v in val.split(";"):
            s = v.split("->")
            if len(s) != 2:
                raise ValueError(f"Malformed context map: {v}")
            if s[0] == "cpu" or s[0].startswith("cuda") or s[0].startswith("opencl"):
                raise ValueError(f"Cannot use {s[0]} as context name")
        return val


def parse_config_string(config_string, issue_warnings=True):
    """
    Parses a config string (comma-separated key=value components) into a dict.
    """
    config_dict = {}
    my_splitter = shlex.shlex(config_string, posix=True)
    my_splitter.whitespace = ","
    my_splitter.whitespace_split = True
    for kv_pair in my_splitter:
        kv_pair = kv_pair.strip()
        if not kv_pair:
            continue
        kv_tuple = kv_pair.split("=", 1)
        if len(kv_tuple) == 1:
            if issue_warnings:
                AesaraConfigWarning.warn(
                    f"Config key '{kv_tuple[0]}' has no value, ignoring it",
                    stacklevel=1,
                )
        else:
            k, v = kv_tuple
            # subsequent values for k will override earlier ones
            config_dict[k] = v
    return config_dict


def config_files_from_aesararc():
    """
    AESARARC can contain a colon-delimited list of config files, like

        AESARARC=~/.aesararc:/etc/.aesararc

    In that case, definitions in files on the right (here, ``~/.aesararc``)
    have precedence over those in files on the left.
    """
    rval = [
        os.path.expanduser(s)
        for s in os.getenv("AESARARC", "~/.aesararc").split(os.pathsep)
    ]
    if os.getenv("AESARARC") is None and sys.platform == "win32":
        # to don't need to change the filename and make it open easily
        rval.append(os.path.expanduser("~/.aesararc.txt"))
    return rval


def _create_default_config():
    # The AESARA_FLAGS environment variable should be a list of comma-separated
    # [section__]option=value entries. If the section part is omitted, there should
    # be only one section that contains the given option.
    AESARA_FLAGS = os.getenv("AESARA_FLAGS", "")
    AESARA_FLAGS_DICT = parse_config_string(AESARA_FLAGS, issue_warnings=True)

    config_files = config_files_from_aesararc()
    aesara_cfg = ConfigParser(
        {
            "USER": os.getenv("USER", os.path.split(os.path.expanduser("~"))[-1]),
            "LSCRATCH": os.getenv("LSCRATCH", ""),
            "TMPDIR": os.getenv("TMPDIR", ""),
            "TEMP": os.getenv("TEMP", ""),
            "TMP": os.getenv("TMP", ""),
            "PID": str(os.getpid()),
        }
    )
    aesara_cfg.read(config_files)
    # Having a raw version of the config around as well enables us to pass
    # through config values that contain format strings.
    # The time required to parse the config twice is negligible.
    aesara_raw_cfg = RawConfigParser()
    aesara_raw_cfg.read(config_files)

    # Instances of AesaraConfigParser can have independent current values!
    # But because the properties are assinged to the type, their existence is global.
    config = AesaraConfigParser(
        flags_dict=AESARA_FLAGS_DICT,
        aesara_cfg=aesara_cfg,
        aesara_raw_cfg=aesara_raw_cfg,
    )
    return config


class _ConfigProxy:
    """Like _SectionRedirect this class enables backwards-compatible access to the
    config settings, but raises DeprecationWarnings with instructions to use `aesara.config`.
    """

    def __init__(self, actual):
        _ConfigProxy._actual = actual

    def __getattr__(self, attr):
        if attr == "_actual":
            return _ConfigProxy._actual
        warnings.warn(
            "Accessing config through `aesara.configparser.config` is deprecated. "
            "Use `aesara.config` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self._actual, attr)

    def __setattr__(self, attr, value):
        if attr == "_actual":
            return setattr(_ConfigProxy._actual, attr, value)
        warnings.warn(
            "Accessing config through `aesara.configparser.config` is deprecated. "
            "Use `aesara.config` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return setattr(self._actual, attr, value)


# Create the actual instance of the config. This one should eventually move to
# `configdefaults`:
_config = _create_default_config()

# The old API often imported the default config object from `configparser`.
# These imports/accesses should be replaced with `aesara.config`, so this wraps
# it with warnings:
config = _ConfigProxy(_config)
# We can't alias the methods of the `config` variable above without already
# triggering the warning.  Instead, we wrap the methods of the actual instance
# with warnings:
change_flags = deprecated("Use aesara.config.change_flags instead!")(
    _config.change_flags
)
_config_print = deprecated("Use aesara.config.config_print instead!")(
    _config.config_print
)
