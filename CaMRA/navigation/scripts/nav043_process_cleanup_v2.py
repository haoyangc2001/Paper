"""Bounded process-group cleanup with retained errors and explicit uncertainty."""
import os
import signal
import subprocess


def stop_group(process, *, wait_s=4.):
    result = {'pid':process.pid, 'events':[], 'errors':[],
              'returncode':None, 'group_absent':False, 'cleanup_confirmed':False}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
            result['events'].append({'signal':int(sig), 'sent':True})
        except ProcessLookupError:
            result['events'].append({'signal':int(sig), 'group_absent':True})
        except Exception as error:
            result['errors'].append({'phase':'signal', 'signal':int(sig), 'error':repr(error)})
        try:
            result['returncode'] = process.wait(timeout=wait_s)
        except Exception as error:
            result['errors'].append({'phase':'wait', 'signal':int(sig), 'error':repr(error)})
    try:
        result['returncode'] = process.poll()
    except Exception as error:
        result['errors'].append({'phase':'poll', 'error':repr(error)})
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        result['group_absent'] = True
    except Exception as error:
        result['errors'].append({'phase':'group_readback', 'error':repr(error)})
    result['cleanup_confirmed'] = (result['returncode'] is not None and result['group_absent'])
    return result
