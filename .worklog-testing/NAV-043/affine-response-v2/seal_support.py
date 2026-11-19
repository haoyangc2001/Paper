from datetime import datetime
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import launch_nav043_affine_response_once as launcher
base = launcher.base

suite = ET.parse(launcher.HERE/'final-basic.xml').find('testsuite')
assert suite is not None and int(suite.attrib['tests']) == 20
assert int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
assert (launcher.HERE/'final-check.exit').read_text().strip() == '0'
identities = base.load(launcher.HERE/'identities.json')
assert identities['protected_identities_checked'] == 295 and identities['raw_input_hashes_checked'] == 90
ready = base.load(launcher.READINESS)
assert ready['status'] == 'PREPARED_ONLY'
assert ready['proposal_sha256'] == base.sha(launcher.affine.PROPOSAL)
for path, digest in ready['source_hashes'].items():
    assert base.sha(path) == digest
assert not launcher.affine.APPROVAL.exists()
assert not Path(ready['formal_output']).exists() and not (launcher.HERE/'launch.lock').exists()
with Path(str(launcher.READINESS)+'.sha256').open('x') as stream:
    stream.write(base.sha(launcher.READINESS)+'\n')
base.write_new(launcher.HERE/'sealed-readiness.json', {'at': datetime.now().astimezone().isoformat(),
    'readiness_sha256': base.sha(launcher.READINESS), 'proposal_sha256': ready['proposal_sha256'],
    'source_hashes': ready['source_hashes'], 'basic_checks': 20,
    'new_real_solves': 0, 'new_synthetic_solves': 0, 'approval_present': False,
    'full_saved_output_audit_performed': False, 'NAV043_passed': False, 'G5_passed': False})
print('prepared-only support sealed; no real fit or full saved-output audit')
