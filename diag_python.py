import sys, ssl
print("executable:", sys.executable)
print("version:", sys.version)
paths = ssl.get_default_verify_paths()
print("cafile:", paths.cafile)
print("capath:", paths.capath)
import urllib.request
try:
    urllib.request.urlopen("https://pypi.org", timeout=5)
    print("HTTPS: OK")
except Exception as e:
    print("HTTPS: FAILED —", e)
