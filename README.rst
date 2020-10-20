SCABox Attack Demo
***************************************************************

.. image:: https://api.travis-ci.com/samiBendou/sca-automation.svg?token=LqpGzZ56omzjYoep5ESp&branch=master

Overview
===============================================================

This repository contains a Python GUI application and a library to retrieve side-channel acquisition data from serial
port and perform an attack. It is part of the `SCABox <https://samibendou.github.io/sca_framework/>`_ project.

- Application : attack from serial port to key guess with exports
- Library : tools to perform attack, serial communication and correlation

The application is based on the library and this least is intended to work for any kind of side-channel leakage data and crypto-algorithm.

Features
===============================================================

Library
---------------------------------------------------------------

- Deserialization of acquisition data
- Fast acquisition data exports and imports
- Fast CPA computation and statistics
- Leakage model hypothesis generation
- Leakage signal processing
- Advanced Encryption Standard (AES)

Application
---------------------------------------------------------------

- Automate acquisition and attack
- Provide correlation and leakage visualization
- Export attack and acquisition results and images
- Parametrize the acquisition and visualize performances

Install
===============================================================

To install the repository you must clone the sources from GitHub and install the pip requirements

.. code-block:: shell

    $ git clone https://github.com/samiBendou/sca-automation
    $ cd sca-automation
    $ pip3 install -r requirements.txt

You might alternatively create a venv and install the requirements inside to use the project. 

Compatibility
---------------------------------------------------------------

The project is compatible with Python 3.8 and latter. It is platform independent.

However the it requires Tkinter to be installed in order to use the GUI application.
The instructions my varies according to which system your are using and we encourage
to visit the Tkinter documentation to install it. 

Usage
===============================================================

Library
---------------------------------------------------------------

The library provides a complete API to develop your own application.
In order to get started you can take a look at the examples and the reference.

Application
---------------------------------------------------------------

The GUI application can be started by running the ``main.py`` script

.. code-block:: shell

    $ cd sca-automation/app 
    $ sudo python3 main.py

You might pass arguments to the ``main.py`` script in order parametrize the acquisition from shell.

.. code-block:: shell

    $ sudo python3 main.py 1024 /dev/ttyUSB1 --chunks 8

To get an exhaustive list, please visit the reference

Documentation
===============================================================

The complete documentation of the project is available `here <https://samibendou.github.io/sca-automation/>`_.

More
===============================================================

SCABox is a project on the topic of side-channel analysis.
The goal of SCABox is to provide a cheap and efficient test-bench for side-channel analysis.

To know more about SCABox please visit our `website <https://samibendou.github.io/sca_framework/>`_.
It provides a tutorials and a wiki about side-channel analysis.

SCABox is an open-source project, all the sources are hosted on GitHub

- `IP repository <https://github.com/samiBendou/sca-ip/>`_
- `Acquisition demo <https://github.com/samiBendou/sca-demo-tdc-aes/>`_
- `Attack demo <https://github.com/samiBendou/sca-automation/>`_
- `SCABox website  <https://github.com/samiBendou/sca_framework/>`_