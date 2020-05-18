
* Run tests.

* Update version in setup.py

* Generate wheel and source archive.

```bash
python3 setup.py sdist bdist_wheel
```

* Upload to PyPi

```bash
twine upload dist/auto_diff-$ver*
```

* Make a release on GitHub Web

* Update PKGBUILD version and hash.

* Update SRCINFO

```bash
makepkg --printsrcinfo > .SRCINFO
```

* Upload to AUR.
