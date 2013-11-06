import os
import glob
import unittest
import HTMLTestRunner

# This code will run all tests in directory named test_*.py

TITLE = 'Test Results'
test_file_strings = glob.glob('test_*.py')
module_strings = [str[0:len(str)-3] for str in test_file_strings]
suites = [unittest.defaultTestLoader.loadTestsFromName(str) for str
          in module_strings]
testSuite = unittest.TestSuite(suites)
unittest.TextTestRunner(verbosity=2).run(testSuite)


outfile = open("report.html", "w")
runner = HTMLTestRunner.HTMLTestRunner(
                stream=outfile,
                title=TITLE,
                description='SimPEG Test Report was automatically generated.'
                )

runner.run(testSuite)
outfile.close()

reader = open("report.html", "r")
writer = open("../../docs/api_TestResults.rst", "w")

writer.write('.. _api_TestResults:\n\nTest Results\n============\n\n.. raw:: html\n\n')

go = False
for line in reader:
    skip = False
    if line == '<style type="text/css" media="screen">\n':
        go = True
    elif line == "<div id='ending'>&nbsp;</div>\n":
        go = False
    elif line == '</head>\n':
        skip = True
    elif line == '<h1>'+TITLE+'</h1>\n':
        skip = True
    elif line == '<body>\n':
        skip = True
    if go and not skip:
        writer.write('    '+line)

writer.close()
reader.close()
os.remove("report.html")
