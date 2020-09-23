sca-automation
***************************************************************
![badge](https://api.travis-ci.com/samiBendou/sca-automation.svg?token=LqpGzZ56omzjYoep5ESp&branch=master)

The automation of the attack is made via Python.
We designed a framework that handles all the stages of the side channel attack within a powerful API.
The main tasks addressed are :

- Automation of UART communication with the demo
- UART to CSV data processing
- Correlation and key guess processing from CSV
- Provide graphical data visualization

It allows to directly drive the demo via multiple Python scripts and a custom library.
It binds with our acquisition standard format to allow you write your own C acquisition.
