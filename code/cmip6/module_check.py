# gen_clean_req.py
import ast
import pathlib
import importlib
import sys

# 1. 标准库模块（Python 内置）
if hasattr(sys, 'stdlib_module_names'):
    STDLIB = set(sys.stdlib_module_names)
else:
    STDLIB = {
        'os', 'sys', 'math', 'json', 're', 'datetime', 'collections', 'threading',
        'queue', 'io', 'urllib', 'http', 'socket', 'subprocess', 'glob', 'shutil',
        'pathlib', 'argparse', 'logging', 'warnings', 'traceback', 'functools',
        'itertools', 'operator', 'copy', 'pickle', 'base64', 'hashlib', 'hmac',
        'struct', 'time', 'random', 'csv', 'configparser', 'xml', 'html', 'cgi',
        'http', 'email', 'ssl', 'tempfile', 'gzip', 'zipfile', 'tarfile', 'bz2',
        'lzma', 'sqlite3', 'ctypes', 'mmap', 'select', 'asyncio', 'concurrent',
        'multiprocessing', 'unittest', 'doctest', 'pdb', 'gc', 'weakref', 'abc',
        'types', 'enum', 'dataclasses', 'typing', 'contextlib', 'textwrap', 'string',
        'decimal', 'fractions', 'statistics', 'array', 'heapq', 'bisect', 'calendar',
        'sched', 'queue', 'sched', 'token', 'tokenize', 'ast', 'symtable', 'compileall',
        'dis', 'opcode', 'marshal', 'imp', 'importlib', 'pkgutil', 'modulefinder',
        'runpy', 'zipimport', 'site', 'user', 'platform', 'distutils', 'ensurepip',
        'venv', 'zipapp', 'turtle', 'cmd', 'shlex', 'code', 'codeop', 'pydoc',
        'profile', 'pstats', 'timeit', 'trace', 'cgitb', 'tabnanny', 'pyclbr',
        'pprint', 'reprlib', 'enum', 'graphlib', 'zoneinfo', 'wsgiref', 'urllib3',
        'http', 'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib', 'smtpd',
        'telnetlib', 'uuid', 'netrc', 'ipaddress', 'socketserver', 'xmlrpc',
        'http', 'json', 'plistlib', 'wave', 'chunk', 'colorsys', 'gettext',
        'locale', 'turtle', 'cmd', 'code', 'codeop', 'pydoc', 'webbrowser',
        'antigravity', 'this', 'symtable', 'tabnanny', 'py_compile', 'compileall',
    }

# 2. 常见假包 / 子模块 / 无效导入（从你的列表中提取）
FAKE_OR_SUBMODULE = {
    '__version__', '_api', '_backport', '_base', '_cell_widths', '_compat',
    '_deprecation_warning', '_dists', '_elffile', '_emoji_codes', '_emoji_replace',
    '_envs', '_export_format', '_extension', '_fileno', '_impl', '_implementation',
    '_in_process', '_inputstream', '_internal_utils', '_log_render', '_loop',
    '_macos', '_manylinux', '_openssl', '_palettes', '_parser', '_pick', '_ratio',
    '_re', '_securetransport', '_spinners', '_ssl_constants', '_structures',
    '_timer', '_tokenizer', '_trie', '_types', '_utils', '_version', '_windows',
    '_wrap', 'adapter', 'adapters', 'after', 'align', 'android', 'ansi',
    'ansitowin32', 'api', 'attr', 'auth', 'backports', 'base', 'bdist_wheel',
    'before', 'before_sleep', 'big5freq', 'big5prober', 'bindings', 'box',
    'brotli', 'build', 'cPickle', 'cache', 'candidates', 'cartopy', 'cdsapi',
    'cells', 'cgi', 'chardistribution', 'charsetgroupprober', 'charsetprober',
    'cmaps', 'codec', 'codingstatemachine', 'color', 'color_triplet', 'colorama',
    'colorlog', 'colors', 'columns', 'com', 'compat', 'connection', 'connectionpool',
    'console', 'constants', 'constrain', 'containers', 'contrib', 'control',
    'controller', 'convert', 'cookielib', 'cookies', 'core', 'cp949prober',
    'cryptography', 'ctags', 'cupy', 'dask', 'database', 'default_styles',
    'dirtools', 'distro', 'dl', 'docutils', 'dummy_thread', 'dummy_threading',
    'ee', 'emoji', 'enums', 'envbuild', 'eofs', 'errors', 'escprober', 'escsm',
    'eucjpprober', 'euckrfreq', 'euckrprober', 'euctwfreq', 'euctwprober',
    'exceptions', 'ext', 'factory', 'fallback', 'fields', 'file_cache',
    'file_proxy', 'filelock', 'filepost', 'filewrapper', 'filters',
    'found_candidates', 'gb2312freq', 'gb2312prober', 'genshi', 'global_land_mask',
    'google', 'hebrewprober', 'highlighter', 'hooks', 'html5parser', 'htmlentitydefs',
    'httplib', 'imp', 'in_process', 'initialise', 'intranges', 'ipywidgets',
    'java', 'jisfreq', 'jnius', 'jpcntx', 'jupyter', 'keyring', 'labels',
    'langbulgarianmodel', 'langgreekmodel', 'langhebrewmodel', 'langrussianmodel',
    'langthaimodel', 'langturkishmodel', 'latin1prober', 'licenses', 'live',
    'live_render', 'lmoments3', 'lockfile', 'lxml', 'macosx_libfile', 'markers',
    'markup', 'mbcharsetprober', 'mbcsgroupprober', 'mbcssm', 'measure', 'metadata',
    'models', 'monkey', 'more', 'mpl_toolkits', 'msilib', 'nap', 'ntlm',
    'ordereddict', 'org', 'osgeo', 'pack', 'package_data', 'packages', 'padding',
    'pager', 'palettable', 'palette', 'panel', 'pkginfo', 'poolmanager', 'pretty',
    'progress_bar', 'protocol', 'providers', 'py', 'py34compat', 'py35compat',
    'py36compat', 'py38compat', 'pycwt', 'pymannkendall', 'rasterio', 'recipes',
    'redis_cache', 'region', 'reporters', 'repr', 'req_file', 'req_install',
    'req_set', 'request', 'requirements', 'resolvers', 'resources', 'response',
    'retry', 'rioxarray', 'rule', 'sbcharsetprober', 'sbcsgroupprober',
    'scope', 'screen', 'segment', 'serialize', 'serializer', 'sessions',
    'shapely', 'sjisprober', 'socks', 'sources', 'specifiers', 'sphinx',
    'spinner', 'ssl_', 'ssltransport', 'status', 'status_codes', 'stop',
    'structs', 'structures', 'style', 'styled', 'syntax', 'table', 'tags',
    'terminal_theme', 'text', 'theme', 'timeout', 'toml', 'tornado', 'treebuilders',
    'treewalkers', 'universaldetector', 'unpack', 'upload', 'url', 'urllib3_secure_extra',
    'utf8prober', 'util', 'utils', 'uts46data', 'vendored', 'version', 'wait',
    'wheelfile', 'win32', 'win32api', 'win32com', 'win32con', 'winterm',
    'wrapper', 'wrappers', 'x_user_defined', 'Cython', 'Pillow'
}

# 3. 第三方包映射（修复常见错误名）
THIRD_PARTY_MAP = {
    'scikit_learn': 'scikit-learn',
    'sklearn': 'scikit-learn',
    'skfuzzy': 'scikit-fuzzy',
    'skill_metrics': 'skill-metrics',
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'yaml': 'PyYAML',
    'Cython': 'Cython',
    'IPython': 'ipython',
    'OpenSSL': 'pyOpenSSL',
    'ConfigParser': 'configparser',  # 标准库
    'Queue': 'queue',              # 标准库
    'StringIO': 'io',              # 标准库
}

def extract_imports_from_file(file_path):
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = alias.name.split('.')[0]
                    imports.add(pkg)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pkg = node.module.split('.')[0]
                imports.add(pkg)
    except Exception as e:
        print(f"跳过 {file_path.name}: {e}")
    return imports

# 扫描项目
project_dir = pathlib.Path(r"E:\GEO\pyproject")
all_imports = set()
print("正在扫描 .py 文件...")

for pyfile in project_dir.rglob("*.py"):
    all_imports.update(extract_imports_from_file(pyfile))

# 过滤第三方包
third_party_raw = {pkg for pkg in all_imports if pkg not in STDLIB and pkg not in FAKE_OR_SUBMODULE}

# 修复包名
third_party_fixed = [THIRD_PARTY_MAP.get(pkg, pkg) for pkg in third_party_raw]

# 检查是否已安装
missing = []
print("\n正在检查是否已安装...")
for pkg in sorted(set(third_party_fixed)):
    try:
        importlib.import_module(pkg)
        print(f"已安装: {pkg}")
    except ImportError:
        missing.append(pkg)
        print(f"缺失: {pkg}")

# 生成 requirements.txt
req_file = project_dir / "requirements_clean.txt"
with open(req_file, "w", encoding="utf-8") as f:
    for pkg in sorted(missing):
        f.write(pkg + "\n")

print(f"\n扫描完成！")
print(f"发现第三方包: {len(third_party_fixed)} 个")
print(f"缺失包: {len(missing)} 个")
if missing:
    print("\n请运行以下命令安装：")
    print("pip install " + " ".join(missing))
else:
    print("\n所有第三方包已安装！")

print(f"\n已生成: {req_file}")