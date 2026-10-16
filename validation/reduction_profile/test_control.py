"""Regression for perf's binary FIFO acknowledgement, including its NUL."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        source = cls.root/'control.cc'
        source.write_text('''#include "control.hh"
int main(int argc, char **argv) {
 if(argc!=3)return 2;
 try { Control c(argv[1],argv[2]);c.send("enable");c.send("disable"); }
 catch(const std::exception&){return 1;}
}
''')
        cls.binary = cls.root/'control'
        subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(HERE),str(source),'-o',str(cls.binary)],check=True)
    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
    def invoke(self, ack):
        commands, responses = self.root/'commands', self.root/'responses'
        responses.write_bytes(ack)
        status = subprocess.run([self.binary,commands,responses],timeout=5).returncode
        return status,commands.read_bytes()
    def test_two_null_terminated_acknowledgements(self):
        status, commands = self.invoke(b'ack\n\0ack\n\0')
        self.assertEqual(status,0)
        self.assertEqual(commands,b'enable\ndisable\n')
    def test_missing_null_rejected(self):
        self.assertEqual(self.invoke(b'ack\nack\n')[0],1)
    def test_corrupted_second_ack_rejected(self):
        self.assertEqual(self.invoke(b'ack\n\0bad\n\0')[0],1)
    def test_truncated_second_ack_rejected(self):
        self.assertEqual(self.invoke(b'ack\n\0ack\n')[0],1)
    def test_disabled_control_does_not_open_files(self):
        self.assertEqual(subprocess.run([self.binary,'-','-'],timeout=5).returncode,0)

if __name__ == '__main__': unittest.main()
