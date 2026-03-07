# Copyright 2022 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
import os
import platform
from pathlib import Path
import shutil
import subprocess

import dimod
import numpy


class build_ext_with_args(build_ext):
    """Add compiler-specific compile/link flags."""

    extra_compile_args = {
        'msvc': ['/std:c++17'],
        'unix': ['-std=c++17'],
    }

    extra_link_args = {
        'msvc': [],
        'unix': ['-std=c++17'],
    }

    @staticmethod
    def _cuda_enabled():
        return os.environ.get('DWAVE_SA_ENABLE_CUDA', '').lower() in ('1', 'true', 'yes', 'on')

    @staticmethod
    def _find_nvcc():
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        candidates = []
        if cuda_home:
            candidates.append(Path(cuda_home) / 'bin' / 'nvcc')
        nvcc_path = shutil.which('nvcc')
        if nvcc_path:
            candidates.append(Path(nvcc_path))
        for c in candidates:
            if c and c.exists():
                return str(c)
        return None

    def _compile_cuda_object(self, ext, source):
        nvcc = self._find_nvcc()
        if nvcc is None:
            raise RuntimeError(
                "DWAVE_SA_ENABLE_CUDA is set, but nvcc was not found. "
                "Set CUDA_HOME/CUDA_PATH or add nvcc to PATH."
            )

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        obj_path = build_temp / (Path(source).name + '.o')

        include_args = []
        for inc in ext.include_dirs or []:
            include_args.extend(['-I', inc])

        cmd = [
            nvcc,
            '-c',
            source,
            '-o',
            str(obj_path),
            '-std=c++17',
            '-O3',
            '--compiler-options',
            '-fPIC',
        ] + include_args

        subprocess.check_call(cmd)
        return str(obj_path)

    def build_extension(self, ext):
        if ext.name == 'dwave.samplers.sa.simulated_annealing' and self._cuda_enabled():
            cuda_source = 'dwave/samplers/sa/src/gpu_sa.cu'
            cuda_object = self._compile_cuda_object(ext, cuda_source)
            ext.extra_objects = list((getattr(ext, 'extra_objects', None) or [])) + [cuda_object]
            ext.define_macros = list((getattr(ext, 'define_macros', None) or [])) + [('DWAVE_SA_WITH_CUDA', '1')]

            cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
            if cuda_home:
                lib64 = str(Path(cuda_home) / 'lib64')
                ext.library_dirs = list((getattr(ext, 'library_dirs', None) or [])) + [lib64]
            ext.libraries = list((getattr(ext, 'libraries', None) or [])) + ['cudart']

        super().build_extension(ext)

    def build_extensions(self):
        compiler = self.compiler.compiler_type

        compile_args = self.extra_compile_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args = list(compile_args)

        link_args = self.extra_link_args[compiler]
        for ext in self.extensions:
            ext.extra_link_args = list(link_args)

        for ext in self.extensions:
            if ext.name == 'dwave.samplers.sa.simulated_annealing':
                if compiler == 'msvc':
                    ext.extra_compile_args.append('/openmp')
                elif compiler == 'unix' and platform.system() != 'Darwin':
                    ext.extra_compile_args.append('-fopenmp')
                    ext.extra_link_args.append('-fopenmp')

        super().build_extensions()


setup(
    cmdclass={'build_ext': build_ext_with_args},
    ext_modules=cythonize([
        Extension('dwave.samplers.greedy.descent', ['dwave/samplers/greedy/descent.pyx']),
        Extension('dwave.samplers.random.cyrandom', ['dwave/samplers/random/cyrandom.pyx']),
        Extension('dwave.samplers.sa.simulated_annealing', ['dwave/samplers/sa/simulated_annealing.pyx']),
        Extension('dwave.samplers.sqa.pimc_annealing', ['dwave/samplers/sqa/pimc_annealing.pyx']),
        Extension('dwave.samplers.sqa.rotormc_annealing', ['dwave/samplers/sqa/rotormc_annealing.pyx']),
        Extension('dwave.samplers.tabu.tabu_search', ['dwave/samplers/tabu/tabu_search.pyx']),
        Extension('dwave.samplers.tree.sample', ['dwave/samplers/tree/sample.pyx']),
        Extension('dwave.samplers.tree.solve', ['dwave/samplers/tree/solve.pyx']),
        Extension('dwave.samplers.tree.utilities', ['dwave/samplers/tree/utilities.pyx']),
    ]),
    include_dirs=[
        dimod.get_include(),
        numpy.get_include(),
    ],
)
