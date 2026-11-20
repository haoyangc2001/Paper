"""Render PREPARED_ONLY support manifest after source-matched basic checks."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
EV = ROOT/'.worklog-testing/NAV-043/motion-candidate-v3'
proposal = NAV/'config/nav043_supported_response_proposal_v3.json'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert (EV/'basic.exit').read_text().strip() == '0'
assert 'Ran 13 tests' in (EV/'basic.log').read_text()
assert json.loads((EV/'support.json').read_text())['support_pass']
assert not (EV/'launch.lock').exists()
source_names = [
    'CaMRA/navigation/scripts/nav043_supported_response.py',
    'CaMRA/navigation/scripts/launch_nav043_supported_response_once.py',
    'CaMRA/navigation/tests/test_supported_response_v3.py',
    '.worklog-testing/NAV-043/motion-candidate-v3/support.py',
    '.worklog-testing/NAV-043/motion-candidate-v3/support.sh',
    '.worklog-testing/NAV-043/motion-candidate-v3/support.json',
    '.worklog-testing/NAV-043/motion-candidate-v3/basic.sh',
    '.worklog-testing/NAV-043/motion-candidate-v3/basic.log',
    '.worklog-testing/NAV-043/motion-candidate-v3/basic.exit']
value = dict(schema='nav043-supported-response-readiness-v3', status='PREPARED_ONLY',
    proposal_sha256=sha(proposal), main_agent_basic_checks=13, basic_pane='%706',
    rank_only_pane='%705', actual_physical_coefficient_solves=0, synthetic_axis_solves=3,
    applications=0, new_episodes=0, W_queries=0, actor_critic_updates=0,
    production_enabled=False, original_recovery_HELD_preserved=True,
    source_hashes={p:sha(ROOT/p) for p in source_names})
path = NAV/'config/nav043_supported_response_readiness_v3.json'
assert not path.exists()
body = json.dumps(value,ensure_ascii=False,indent=2)+'\n'
print('*** Begin Patch')
for target, text in [(path,body),(Path(str(path)+'.sha256'),hashlib.sha256(body.encode()).hexdigest()+'\n')]:
    print('*** Add File: '+str(target.relative_to(ROOT)))
    for line in text.splitlines():print('+'+line)
print('*** End Patch')
