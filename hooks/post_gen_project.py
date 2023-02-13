import os
import sys
import shutil
import subprocess

repo_name = '{{ cookiecutter.repository_name }}'
repo_url = 'git@github.com:VisiumCH/' + repo_name + '.git'

subprocess.call(['git', 'init'])

subprocess.call(['git', 'config', 'user.name', 'Cookiecutter'])
subprocess.call(['git', 'config', 'user.email', 'cookiecutter@visium.ch'])

subprocess.call(['git', 'remote', 'add', 'origin', repo_url])
subprocess.call(['git', 'add', '-A'])
subprocess.call(['git', 'commit', '-m', 'Initalization'])
subprocess.call(['git', 'push', '-f', 'origin', 'main'])

subprocess.call(['git', 'config', '--unset', 'user.name'])
subprocess.call(['git', 'config', '--unset', 'user.email'])

os.chdir('..')
shutil.rmtree(repo_name)
