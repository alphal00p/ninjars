"""Test nested diagnostic accounting independently of any physics runtime."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent

class DiagnosticTests(unittest.TestCase):
    def test_counts_rows_and_nested_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'main.rs'; binary=root/'probe'
            source.write_text(f'''#[path={json.dumps(str(HERE/'diagnostic.rs'))}] mod diagnostic;
fn main() {{
 {{ let _off=diagnostic::Span::phase("disabled",99); }}
 diagnostic::start();
 {{ let _outer=diagnostic::Span::phase("outer",3);
   {{ let _child=diagnostic::Span::kernel(6,7,8); std::hint::black_box(7); }}
   {{ let _child=diagnostic::Span::kernel(6,7,8); std::hint::black_box(8); }}
 }}
 print!("{{}}",diagnostic::stop());
 diagnostic::start();
 print!("{{}}",diagnostic::stop());
}}''')
            subprocess.run(['rustc','--edition=2024',source,'-o',binary],check=True)
            first,second=map(json.loads,subprocess.check_output([binary],text=True).splitlines())
            self.assertEqual(second['records'],[])
            phases=[r for r in first['records'] if r['kind']=='phase']
            kernels=[r for r in first['records'] if r['kind']=='kernel']
            self.assertEqual(len(phases),1)
            self.assertEqual((phases[0]['calls'],phases[0]['rows']),(1,3))
            self.assertEqual(sum(r['calls'] for r in kernels),2)
            self.assertEqual(sum(r['rows'] for r in kernels),12)
            self.assertTrue(all(r['parent']=='outer' and r['inputs']==7 and r['outputs']==8 for r in kernels))
            self.assertEqual(phases[0]['inclusive_ns']-phases[0]['exclusive_ns'],sum(r['inclusive_ns'] for r in kernels))
            self.assertTrue(all(r['inclusive_ns']==r['exclusive_ns'] for r in kernels))
            self.assertFalse(first['performance_acceptance'])

if __name__ == '__main__': unittest.main()
