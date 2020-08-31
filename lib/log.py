"""Python-agnostic SCA data parsing, logging and reporting module.

This module is designed as a class library providing *entity classes*
to represent the side channel data acquired from SoC.

It provides binary data parsing in order to read
acquisition data from a serial source or a binary file and
retrieve it in the entity classes.

Examples
--------
>>> from lib import log
>>> s = log.read.serial("com5", 256, "hw", False)
>>> s = log.read.file("path/to/binary/file")
>>> parser = log.Parser.from_bytes(s)

All the entity classes of the module provide CSV support
to allow processing acquisition data without parsing.
It also provides formatting data in a more human-readable format.

Examples
--------
>>> from lib import log
>>> meta = log.Meta.from_csv("path/to/meta.csv")
>>> leak = log.Leak.from_csv("path/to/leak.csv")
>>> # some processing on meta
>>> meta.to_csv("path/to/meta.csv")

"""

import csv
from warnings import warn
from itertools import zip_longest
import serial

from lib.utils import decode_hamming, check_hex

START_TRACE = b"\xfe\xfe\xfe\xfe"  # Start traces tag
END_ACQ = b"\xff\xff\xff\xff"  # End acquisition tag
ACQ_CMD = "sca"


class read:
    """Namespace for binary read operations.

    """

    @classmethod
    def file(cls, log_path) -> bytes:
        """Reads binary data from file.

        Parameters
        ----------
        log_path : str
            Path to the file to read.
        Returns
        -------
        bytes
            Binary content of the file.
        """
        with open(log_path, "rb") as log_file:
            s = log_file.read()
        return s

    @classmethod
    def serial(cls, port, iterations, mode, inv, verbose=False):
        """Launches acquisition and reads data from serial port.

        This method sends the side-channel acquisition command
        to the SoC and then reads the serial output.

        Parameters
        ----------
        port : str
            Serial port to read.
        iterations : int
            Requested count of traces.
        mode : str
            Encryption mode.
        inv : bool
            True if encryption direction is decrypt.
        verbose :
            If false, the traces data will be compressed using
            hamming weight encoding.

        Returns
        -------
        bytes
            Binary data from the serial port.

        See Also
        --------
        py_sca.lib.utils.decode_hamming : Decode hamming weight.

        """
        cmd = (ACQ_CMD,
               " -t %d" % iterations,
               " -v" if verbose else "",
               " -h" if mode == "hw" else "",
               " -i" if inv else ""
               )
        with serial.Serial(port, 921_600, parity=serial.PARITY_NONE, xonxoff=False) as ser:
            ser.flush()
            ser.write(("%s%s%s%s%s\n" % cmd).encode())
            s = ser.read_all()
            while s[-8:].find(END_ACQ) == -1:
                while ser.in_waiting == 0:
                    continue
                while ser.in_waiting != 0:
                    s += ser.read_all()
        return s


class write:
    """Namespace for binary write operations.

    """

    @classmethod
    def bytes(cls, s, path):
        """Writes binary data to a file.

        Parameters
        ----------
        s : bytes
            Binary data to write.
        path :
            Path to file.

        """
        with open(path, "wb+") as log_file:
            log_file.write(s)


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
    KEY = "key"
    PLAIN = "plain"
    CIPHER = "cipher"
    SAMPLES = "samples"
    CODE = "code"
    WEIGHTS = "weights"
    DELIMITER = b":"

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


class Data:
    """Encryption channel data.

    This class is designed to represent AES encryption data
    for each trace acquired.

    data are represented as 32-bit hexadecimal strings
    in order to accelerate IO operations on the encrypted block.

    Attributes
    ----------
    plains: list[list[str]]
        Plains words for each trace.
    ciphers: list[list[str]]
        Cipher words for each trace.
    keys: list[list[str]]
        Keys words for each trace.

    Raises
    ------
    ValueError
        If the three list attributes does not have the same length.

    """

    def __init__(self, plains=None, ciphers=None, keys=None):
        """Initializes an object by giving attributes values

        Parameters
        ----------
        plains: list[list[str]], optional
            Plains words for each trace.
        ciphers: list[list[str]], optional
            Cipher words for each trace.
        keys: list[list[str]], optional
            Keys words for each trace.

        """
        self.plains = plains or []
        self.ciphers = ciphers or []
        self.keys = keys or []

        if len(self.plains) != len(self.ciphers) or len(self.plains) != len(self.keys):
            raise ValueError("Plains, ciphers and keys must have the same length")

    def clear(self):
        """Clears all the attributes.

        """
        self.plains.clear()
        self.ciphers.clear()
        self.keys.clear()

    def to_csv(self, path):
        """Exports encryption data to a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file to write.

        """
        with open(path, "w", newline="") as file:
            header = ["p0", "p1", "p2", "p3", "c0", "c1", "c2", "c3", "k0", "k1", "k2", "k3"]
            writer = csv.writer(file, delimiter=";")
            writer.writerow(header)
            for plain, cipher, key in zip_longest(self.plains, self.ciphers, self.keys, fillvalue=self.keys[0]):
                writer.writerow(plain + cipher + key)

    @classmethod
    def from_csv(cls, path):
        """Imports encryption data from CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV to read.

        Returns
        -------
        Data
            Imported encryption data.
        """
        plains = []
        ciphers = []
        keys = []
        with open(path, "r", newline="") as file:
            reader = csv.reader(file, delimiter=";")
            next(reader)
            for row in reader:
                if not row:
                    continue
                plains.append(list(row[0:4]))
                ciphers.append(list(row[4:8]))
                keys.append(list(row[8:12]))

        return Data(plains, ciphers, keys)


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
            writer = csv.writer(file, delimiter=";")
            rows = self.__dict__
            header = list(rows.keys())
            values = list(rows.values())
            writer.writerows([header, values])

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
            reader = csv.reader(file, delimiter=";")
            try:
                next(reader)
            except StopIteration:
                return None
            row = next(reader)
        return Meta(row[0], row[1], int(row[2]), int(row[3]), int(row[4]), int(row[5]))


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

    def __init__(self, traces=None):
        """Parses raw traces data.

        Parameters
        ----------
        traces : list[list[int]], optional
            Power consumption leakage signal for each acquisition.

        """
        if traces:
            self.samples = list(map(len, traces)) if traces else []
            self.traces = traces or []
        else:
            self.samples = []
            self.traces = []

    def clear(self):
        """Clears leakage data.

        """
        self.samples.clear()
        self.traces.clear()

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
        leaks = []
        with open(path, "r", newline="") as file:
            reader = csv.reader(file, delimiter=";")
            for row in reader:
                if not row:
                    continue
                leaks.append(list(map(lambda x: int(x), row)))
        return Leak(leaks)


class Parser:
    """Binary data parser.

    This class is designed to parse binary acquisition data
    and store parsed data in the entity classes in order
    to later import and export it.

    Attributes
    ----------
    leak : Leak
        Leakage data.
    data : Data
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
        data : Data
            Encryption data.
        meta : Meta
            Meta-data.

        """
        self.leak = leak or Leak()
        self.data = data or Data()
        self.meta = meta or Meta()

    @classmethod
    def from_bytes(cls, s, inv=False):
        """Initializes an object with binary data.
        
        Parameters
        ----------
        s : bytes
            Binary data.
        inv : bool
            True if encryption direction is decrypt.
        Returns
        -------
        Parser
            Parser initialized with data.

        """
        return Parser().parse_bytes(s, inv)

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
            self.data.keys, self.data.plains, self.data.ciphers, self.leak.samples,
            self.leak.traces
        ]))
        n_min = min(lens)
        n_max = max(lens)

        if n_max == n_min:
            self.leak.samples.pop()
            self.leak.traces.pop()
            self.data.keys.pop()
            self.data.plains.pop()
            self.data.ciphers.pop()
            return

        while len(self.leak.samples) != n_min:
            self.leak.samples.pop()
        while len(self.leak.traces) != n_min:
            self.leak.traces.pop()
        while len(self.data.keys) != n_min:
            self.data.keys.pop()
        while len(self.data.plains) != n_min:
            self.data.plains.pop()
        while len(self.data.ciphers) != n_min:
            self.data.ciphers.pop()

        return self

    def clear(self):
        """Clears all the parser data.

        """
        self.leak.clear()
        self.meta.clear()
        self.data.clear()

    def parse_bytes(self, s, inv=False):
        """Parses the given bytes to retrieve acquisition data.

        If inv`` is not specified the parser will infer the
        encryption direction from ``s``.

        Parameters
        ----------
        s : bytes
            Binary data.
        inv : bool
            True if encryption direction is decrypt.

        Returns
        -------
        Parser
            Reference to self.
        """
        keywords = Keywords(inv=inv)
        expected = next(keywords)
        valid = True
        lines = s.split(b"\r\n")
        for idx, line in enumerate(lines):
            if valid is False:
                valid = line == START_TRACE
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
        self.meta.iterations += len(self.leak.traces)
        return self

    def __decode_hamming(self, c):
        return decode_hamming(c, self.meta.offset)

    def __parse_line(self, line, expected):
        if line in (END_ACQ, START_TRACE):
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
            getattr(self.data, keyword + "s").append(list(map(check_hex, data.split(b" "))))
        elif keyword == Keywords.SAMPLES:
            self.leak.samples.append(int(data))
        elif keyword == Keywords.CODE:
            self.leak.traces.append(list(map(self.__decode_hamming, line[6:])))
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
