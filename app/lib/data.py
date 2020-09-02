"""Python-agnostic SCA data parsing, logging and reporting module.

This module is designed as a class library providing *entity classes*
to represent the side channel data acquired from SoC.

It provides binary data parsing in order to read
acquisition data from a serial sources or a binary file and
retrieve it in the entity classes.

Examples
--------

>>> from lib import read
>>> s = read.source("/dev/ttyUSB1", 256, "hw", False)
>>> parser = log.Parser.from_bytes(s)

>>> from lib import read
>>> s = read.file("path/to/binary/file")
>>> parser = log.Parser.from_bytes(s)

All the entity classes of the module provide CSV support
to allow processing acquisition data without parsing.
It also provides formatting data in a more human-readable format.

Examples
--------
>>> from lib import data
>>> meta = data.Meta.from_csv("path/to/meta.csv")
>>> leak = data.Leak.from_csv("path/to/leak.csv")
>>> # some processing on meta
>>> meta.to_csv("path/to/meta.csv")

"""

import csv
from warnings import warn
from lib.utils import decode_hamming, check_hex


class Request:
    """Data processing request.

    This class provides a simple abstraction to wrap
    file naming arguments during acquisition, import or export.

    The these arguments combined together form a *data request*
    specifying all the characteristics of the target data-set.

    Attributes
    ----------
    iterations : int
        Requested count of traces.
    mode : str
        Encryption mode.
    direction : str
        Encrypt direction.
    source : bool
        True if an acquisition from serial port is requested.
    """

    ACQ_CMD_NAME = "sca"

    class Modes:
        HARDWARE = "hw"
        SOFTWARE = "sw"

    class Directions:
        ENCRYPT = "enc"
        DECRYPT = "dec"

    class Sources:
        FILE = "file"
        SERIAL = "serial"

    def __init__(self, iterations,
                 source=Sources.FILE,
                 direction=Directions.ENCRYPT,
                 mode=Modes.HARDWARE,
                 verbose=False):
        """Initializes a request with attributes.

        Attributes
        ----------
        iterations : int
            Requested count of traces.
        source : str, optional
            Acquisition source.
        mode : str, optional
            Encryption mode.
        inverse : bool, optional
            True if encryption direction is decrypt.
        """
        self.source = source
        self.iterations = iterations
        self.mode = mode
        self.direction = direction
        self.verbose = verbose

    @classmethod
    def from_args(cls, args):
        """Initializes a request with a previously parsed command.

        Parameters
        ----------
        args
            Parsed arguments.

        """
        ret = Request(args.iterations)
        if hasattr(args, "source"):
            ret.source = args.source
        if hasattr(args, "mode"):
            ret.mode = args.mode
        if hasattr(args, "direction"):
            ret.direction = args.direction
        if hasattr(args, "verbose"):
            ret.verbose = args.verbose

        return ret

    @classmethod
    def from_meta(cls, meta):
        """Initializes a request with command line arguments.

        Parameters
        ----------
        meta : sca-automation.lib.log.Meta
            Meta-data.

        """
        return Request(meta.iterations, mode=meta.mode, direction=meta.direction)

    def filename(self, prefix, suffix=""):
        """Creates a filename based on the request.

        This method allows to consistently name the files
        according to request's characteristics.

        Parameters
        ----------
        prefix : str
            Prefix of the filename.
        suffix : str, optional
            Suffix of the filename.

        Returns
        -------
        str :
            The complete filename.

        """
        return f"{prefix}_{self.mode}_{self.direction}_{self.iterations}{suffix}"

    def command(self, name):
        return "{}{}{}{}{}".format(name,
                                   " -t %d" % self.iterations,
                                   " -v" if self.verbose else "",
                                   " -h" if self.mode == Request.Modes.HARDWARE else "",
                                   " -i" if self.direction == Request.Directions.DECRYPT else "")


class Keywords:
    """Iterates over the binary log keywords.

    This iterator allows to represent the keywords and
    the order in which they appear in the binary log consistently.

    This feature is useful when parsing log lines in order
    to avoid inserting a sequence that came in the wrong order.

    At each iteration the next expected keyword is returned.

    ``meta`` attribute allows to avoid expect again meta keywords
    when an error occurs and the iterator must be reset.

    Attributes
    ----------
    idx: int
        Current keyword index.
    meta: bool
        True if the meta-data keyword have already been browsed.
    inv: bool
        True if the keywords follow the inverse encryption sequence.
    metawords: str
        Keywords displayed once at the beginning of acquisition.
    datawords: str
        Keywords displayed recurrently during each trace acquisition.
    value: str
        Value of the current keyword.

    """

    MODE = "mode"
    DIRECTION = "direction"
    SENSORS = "sensors"
    TARGET = "target"
    KEY = "keys"
    PLAIN = "plains"
    CIPHER = "ciphers"
    SAMPLES = "samples"
    CODE = "code"
    WEIGHTS = "weights"
    OFFSET = "offset"
    ITERATIONS = "iterations"

    DELIMITER = b":"
    START_TRACE_TAG = b"\xfe\xfe\xfe\xfe"
    END_ACQ_TAG = b"\xff\xff\xff\xff"

    def __init__(self, meta=False, inv=False, verbose=False):
        """Initializes a new keyword iterator.

        Parameters
        ----------
        meta : bool
            If true, meta-data keywords will be ignored.
        inv: bool
            True if the keywords follow the inverse encryption sequence.
        """
        self.idx = 0
        self.meta = meta
        self.value = ""
        if not meta:
            self.__build_metawords()
        self.__build_datawords(inv, verbose)
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        current = self.value
        if self.meta:
            self.idx = (self.idx + 1) % len(self.datawords)
            self.value = self.datawords[self.idx]
            return current

        if self.idx < len(self.metawords):
            self.idx += 1

        if self.idx == len(self.metawords):
            self.reset(meta=True)
            return current

        self.value = self.metawords[self.idx]
        return current

    def reset(self, meta=None):
        self.idx = 0
        self.meta = meta or self.meta
        self.value = self.datawords[0] if self.meta else self.metawords[0]

    def __build_metawords(self):
        self.metawords = [Keywords.SENSORS, Keywords.TARGET, Keywords.MODE, Keywords.DIRECTION, Keywords.KEY]

    def __build_datawords(self, inv, verbose):
        self.datawords = [Keywords.CIPHER, Keywords.PLAIN] if inv else [Keywords.PLAIN, Keywords.CIPHER]
        self.datawords += [Keywords.SAMPLES, Keywords.WEIGHTS] if verbose else [Keywords.SAMPLES, Keywords.CODE]


class Channel:
    """Encryption channel data.

    This class is designed to represent AES 128 encryption data
    for each trace acquired.

    data are represented as 32-bit hexadecimal strings
    in order to accelerate IO operations on the encrypted block.

    Attributes
    ----------
    plains: list[str]
        Hexadecimal plain data of each trace.
    ciphers: list[str]
        Hexadecimal cipher data of each trace.
    keys: list[str]
        Hexadecimal key data of each trace.

    Raises
    ------
    ValueError
        If the three list attributes does not have the same length.

    """

    STR_MAX_LINES = 32

    def __init__(self, plains=None, ciphers=None, keys=None):
        """Initializes an object by giving attributes values

        Parameters
        ----------
        plains: list[str], optional
            Plains words for each trace.
        ciphers: list[str], optional
            Cipher words for each trace.
        keys: list[str], optional
            Keys words for each trace.

        """
        self.plains = plains or []
        self.ciphers = ciphers or []
        self.keys = keys or []

        if len(self.plains) != len(self.ciphers) or len(self.plains) != len(self.keys):
            raise ValueError("Inconsistent plains, cipher and keys length")

    def __getitem__(self, item):
        return self.plains[item], self.ciphers[item], self.keys[item]

    def __setitem__(self, item, value):
        plain, cipher, key = value
        self.plains[item] = plain
        self.ciphers[item] = cipher
        self.keys[item] = key

    def __delitem__(self, item):
        del self.plains[item]
        del self.ciphers[item]
        del self.keys[item]

    def __len__(self):
        return len(self.plains)

    def __iter__(self):
        return zip(self.plains, self.ciphers, self.keys)

    def __reversed__(self):
        return zip(reversed(self.plains), reversed(self.ciphers), reversed(self.keys))

    def __repr__(self):
        return f"{type(self).__name__}({self.plains!r}, {self.ciphers!r}, {self.keys!r})"

    def __str__(self):
        n = len(self.plains[0]) + 4
        ret = f"{'plains':<{n}s}{'ciphers':<{n}s}{'keys':<{n}s}"
        for d, (plain, cipher, key) in enumerate(self):
            if d == Channel.STR_MAX_LINES:
                return ret + f"\n{len(self) - d} more..."
            ret += f"\n{plain:<{n}s}{cipher:<{n}s}{key:<{n}s}"
        return ret

    def __iadd__(self, other):
        self.plains += other.plains
        self.ciphers += other.ciphers
        self.keys += other.keys
        return self

    def clear(self):
        """Clears all the attributes.

        """
        self.plains.clear()
        self.ciphers.clear()
        self.keys.clear()

    def append(self, item):
        plain, cipher, key = item
        self.plains.append(plain)
        self.ciphers.append(cipher)
        self.keys.append(key)

    def pop(self):
        self.plains.pop()
        self.ciphers.pop()
        self.keys.pop()

    def to_csv(self, path):
        """Exports encryption data to a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file to write.

        """
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, [Keywords.PLAIN, Keywords.CIPHER, Keywords.KEY], delimiter=";")
            writer.writeheader()
            for plain, cipher, key in self:
                writer.writerow({Keywords.PLAIN: plain,
                                 Keywords.CIPHER: cipher,
                                 Keywords.KEY: key})

    @classmethod
    def from_csv(cls, path):
        """Imports encryption data from CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV to read.

        Returns
        -------
        Channel
            Imported encryption data.
        """
        plains = []
        ciphers = []
        keys = []
        with open(path, "r", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            for row in reader:
                plains.append(row[Keywords.PLAIN])
                ciphers.append(row[Keywords.CIPHER])
                keys.append(row[Keywords.KEY])

        return Channel(plains, ciphers, keys)


class Meta:
    """Meta-data of acquisition.

   This class is designed to represent additional infos
   about the current side-channel acquisition run.

    Attributes
    ----------
    mode : str
        Encryption mode.
    direction : str
        Encryption direction, either encrypt or decrypt.
    target : int
        Sensors calibration target value.
    sensors : int
        Count of sensors.
    iterations : int
        Requested count of traces.
    offset : int
        If the traces are ASCII encoded, code offset
    """

    def __init__(self, mode=None, direction=None, target=0, sensors=0, iterations=0, offset=0):
        """Construct an object with given meta-data

        Parameters
        ----------
        mode : str, optional
            Encryption mode.
        direction : str, optional
            Encryption direction, either encrypt or decrypt.
        target : int, optional
            Sensors calibration target value.
        sensors : int, optional
            Count of sensors.
        iterations : int, optional
            Requested count of traces.
        offset : int, optional
            If the traces are ASCII encoded, code offset.
        """
        self.mode = mode
        self.direction = direction
        self.target = target
        self.sensors = sensors
        self.iterations = iterations
        self.offset = offset

    def __repr__(self):
        return f"{type(self).__name__}(" \
               f"{self.mode!r}, " \
               f"{self.direction!r}, " \
               f"{self.target!r}, " \
               f"{self.sensors!r}, " \
               f"{self.iterations!r}, " \
               f"{self.offset!r})"

    def __str__(self):
        dl = str(Keywords.DELIMITER, 'ascii')
        return f"{Keywords.MODE}{dl} {self.mode}\n" \
               f"{Keywords.DIRECTION}{dl} {self.direction}\n" \
               f"{Keywords.TARGET}{dl} {self.target}\n" \
               f"{Keywords.SENSORS}{dl} {self.sensors}\n" \
               f"{Keywords.ITERATIONS}{dl} {self.iterations}\n" \
               f"{Keywords.OFFSET}{dl} {self.offset}"

    def clear(self):
        """Resets meta-data.

        """
        self.mode = None
        self.direction = None
        self.target = 0
        self.sensors = 0
        self.iterations = 0
        self.offset = 0

    def to_csv(self, path):
        """Exports meta-data to a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file to write.

        """
        with open(path, "w", newline="") as file:
            fieldnames = [Keywords.MODE,
                          Keywords.DIRECTION,
                          Keywords.TARGET,
                          Keywords.SENSORS,
                          Keywords.ITERATIONS,
                          Keywords.OFFSET]
            writer = csv.DictWriter(file, fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerow({Keywords.MODE: self.mode,
                             Keywords.DIRECTION: self.direction,
                             Keywords.TARGET: self.target,
                             Keywords.SENSORS: self.sensors,
                             Keywords.ITERATIONS: self.iterations,
                             Keywords.OFFSET: self.offset})

    @classmethod
    def from_csv(cls, path):
        """Imports meta-data from CSV file.

        If the file is empty returns an empty meta data object.

        Parameters
        ----------
        path : str
            Path to the CSV to read.

        Returns
        -------
        Meta
            Imported meta-data.
        """
        with open(path, "r", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            try:
                row = next(reader)
            except StopIteration:
                return None
        return Meta(row[Keywords.MODE],
                    row[Keywords.DIRECTION],
                    int(row[Keywords.TARGET]),
                    int(row[Keywords.SENSORS]),
                    int(row[Keywords.ITERATIONS]),
                    int(row[Keywords.OFFSET]))


class Leak:
    """Side-channel leakage data.

    This class represents the power consumption traces
    acquired during encryption in order to process these.

    Attributes
    ----------
    samples: list[int]
        Count of samples for each traces.
    traces: list[list[int]]
        Power consumption leakage signal for each acquisition.

    """
    STR_MAX_LINES = 32

    def __init__(self, traces=None):
        """Parses raw traces data.

        Parameters
        ----------
        traces : list[list[int]], optional
            Power consumption leakage signal for each acquisition.

        """
        self.samples = list(map(len, traces)) if traces else []
        self.traces = traces or []

    def __getitem__(self, item):
        return self.traces[item]

    def __setitem__(self, item, value):
        self.traces[item] = value
        self.samples[item] = len(value)

    def __delitem__(self, item):
        del self.traces[item]
        del self.samples[item]

    def __len__(self):
        return len(self.traces)

    def __iter__(self):
        return iter(self.traces)

    def __reversed__(self):
        return iter(reversed(self.traces))

    def __repr__(self):
        return f"{type(self).__name__}({self.traces!r})"

    def __str__(self):
        ret = f"{'no.':<16s}traces"
        for d, trace in enumerate(self):
            if d == Leak.STR_MAX_LINES:
                return ret + f"\n{len(self) - d} more..."
            ret += f"\n{d:<16d}"
            for t in trace:
                ret += f"{t:<4d}"
        return ret

    def __iadd__(self, other):
        self.traces += other.traces
        self.samples += other.samples
        return self

    def clear(self):
        self.traces.clear()
        self.samples.clear()

    def append(self, item):
        self.traces.append(item)
        self.samples.append(len(item))

    def pop(self):
        self.traces.pop()
        self.samples.pop()

    def to_csv(self, path):
        """Exports leakage data to CSV.

        Parameters
        ----------
        path : str
            Path to the CSV file to write.

        """
        with open(path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerows(self.traces)

    @classmethod
    def from_csv(cls, path):
        """Imports leakage data from CSV.

        Parameters
        ----------
        path : str
            Path to the CSV to read.

        Returns
        -------
        Leak
            Imported leak data.
        """
        traces = []
        with open(path, "r", newline="") as file:
            reader = csv.reader(file, delimiter=";")
            for row in reader:
                if not row:
                    continue
                traces.append(list(map(lambda x: int(x), row)))
        return Leak(traces)


class Parser:
    """Binary data parser.

    This class is designed to parse binary acquisition data
    and store parsed data in the entity classes in order
    to later import and export it.

    Attributes
    ----------
    leak : Leak
        Leakage data.
    data : Channel
        Encryption data.
    meta : Meta
        Meta-data.

    See Also
    --------
    Keywords : iterator representing the keyword sequence

    """

    def __init__(self, leak=None, data=None, meta=None):
        """Initializes an object with given attributes

        Parameters
        ----------
        leak : Leak
            Leakage data.
        data : Channel
            Encryption data.
        meta : Meta
            Meta-data.

        """
        self.leak = leak or Leak()
        self.channel = data or Channel()
        self.meta = meta or Meta()

        if len(self.leak) != len(self.channel) or len(self.channel) > self.meta.iterations:
            raise ValueError("Incompatible leaks and data lengths")

    @classmethod
    def from_bytes(cls, s, direction=Request.Directions.ENCRYPT):
        """Initializes an object with binary data.
        
        Parameters
        ----------
        s : bytes
            Binary data.
        direction : bool
            True if encryption direction is decrypt.
        Returns
        -------
        Parser
            Parser initialized with data.

        """
        return Parser().parse_bytes(s, direction)

    def pop(self):
        """Pops acquired value until data lengths matches.

        This method allows to guarantee that the data contained
        in the parser will always have the same length which is
        the number of traces parsed.

        Returns
        -------
        Parser
            Reference to self.
        """
        lens = list(map(len, [
            self.channel.keys, self.channel.plains, self.channel.ciphers, self.leak.samples, self.leak.traces
        ]))
        n_min = min(lens)
        n_max = max(lens)

        if n_max == n_min:
            self.leak.pop()
            self.channel.pop()
            self.meta.iterations -= 1
            return

        while len(self.leak.samples) != n_min:
            self.leak.samples.pop()
        while len(self.leak.traces) != n_min:
            self.leak.traces.pop()
        while len(self.channel.keys) != n_min:
            self.channel.keys.pop()
        while len(self.channel.plains) != n_min:
            self.channel.plains.pop()
        while len(self.channel.ciphers) != n_min:
            self.channel.ciphers.pop()

        return self

    def clear(self):
        """Clears all the parser data.

        """
        self.leak.clear()
        self.meta.clear()
        self.channel.clear()

    def parse_bytes(self, s, direction=Request.Directions.ENCRYPT):
        """Parses the given bytes to retrieve acquisition data.

        If inv`` is not specified the parser will infer the
        encryption direction from ``s``.

        Parameters
        ----------
        s : bytes
            Binary data.
        direction : bool
            True if encryption direction is decrypt.

        Returns
        -------
        Parser
            Reference to self.
        """
        keywords = Keywords(inv=direction == Request.Directions.DECRYPT)
        expected = next(keywords)
        valid = True
        lines = s.split(b"\r\n")
        for idx, line in enumerate(lines):
            if valid is False:
                valid = line == Keywords.START_TRACE_TAG
                continue
            try:
                if self.__parse_line(line, expected):
                    expected = next(keywords)
            except (ValueError, UnicodeDecodeError, RuntimeError) as e:
                args = (e, len(self.leak.traces), idx, line)
                warn("parsing error\nerror: {}\niteration: {:d}\nline {:d}: {}".format(*args))
                keywords.reset(keywords.meta)
                expected = next(keywords)
                valid = False
                self.pop()

        if len(self.channel.keys) == 1:
            self.channel.keys = [self.channel.keys[0]] * len(self.channel)
        self.meta.iterations += len(self.leak.traces)
        return self

    def __decode_hamming(self, c):
        return decode_hamming(c, self.meta.offset)

    def __parse_line(self, line, expected):
        if line in (Keywords.END_ACQ_TAG, Keywords.START_TRACE_TAG):
            return False
        split = line.strip().split(Keywords.DELIMITER)
        try:
            keyword = str(split[0], "ascii").strip()
            data = split[1].strip()
        except IndexError:
            return False

        if keyword != expected:
            raise RuntimeError("expected %s keyword not %s" % (expected, keyword))

        if keyword in (Keywords.MODE, Keywords.DIRECTION):
            setattr(self.meta, keyword, str(data, "ascii"))
        elif keyword in (Keywords.SENSORS, Keywords.TARGET):
            setattr(self.meta, keyword, int(data))
        elif keyword in (Keywords.KEY, Keywords.PLAIN, Keywords.CIPHER):
            getattr(self.channel, keyword).append(check_hex(data.replace(b" ", b"")))
        elif keyword == Keywords.SAMPLES:
            self.leak.samples.append(int(data))
        elif keyword == Keywords.CODE:
            self.leak.traces.append(list(map(self.__decode_hamming, line[(len(Keywords.CODE) + 2):])))
        elif keyword == Keywords.WEIGHTS:
            self.leak.traces.append(list(map(int, data.split(b","))))
        else:
            return False

        if keyword == Keywords.TARGET:
            self.meta.offset = self.meta.sensors * self.meta.target

        if keyword in (Keywords.CODE, Keywords.WEIGHTS):
            n = self.leak.samples[-1]
            m = len(self.leak.traces[-1])
            if m != n:
                raise RuntimeError("trace lengths mismatch %d != %d" % (m, n))

        return True
