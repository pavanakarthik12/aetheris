import ssl
import sys
print("Python:", sys.executable)
print("SSL paths:", ssl.get_default_verify_paths())
try:
    import certifi
    print("certifi:", certifi.where())
except ImportError:
    print("certifi: not installed")

# Try a real HTTPS connection
import urllib.request
try:
    urllib.request.urlopen("https://pypi.org", timeout=5)
    print("HTTPS to pypi.org: OK")
except Exception as e:
    print("HTTPS to pypi.org: FAILED —", e)
