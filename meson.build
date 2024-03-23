project('prima', 'fortran',
    # default_options : ['build_library=shared']
)


fs = import('fs')

fortran_compiler = meson.get_compiler('fortran')

# the compiler enables an executable stack because of nested functions and this is fine
if fortran_compiler.get_id() == 'gcc'
    # link_args = ['-Wl,--no-warn-execstack']
    # add_project_link_arguments(link_args, language : 'fortran')
    have_warn_execstack = fortran_compiler.has_link_argument('-Wl,--no-warn-execstack')
endif

if get_option('prima_heap_arrays')
  if fortran_compiler.get_id() == 'gcc'
    add_project_arguments(['-fno-stack-arrays', '-frecursive'], language : 'fortran')
  elif fortran_compiler.get_id().startswith('intel')
    if fortran_compiler.get_argument_syntax() == 'msvc'
        add_project_arguments(['/heap-arrays', '/assume:recursion'], language : 'fortran')
    else
        add_project_arguments(['-heap-arrays', '-assume recursion'], language : 'fortran')
    endif
  elif fortran_compiler.get_id() == 'nvidia_hpc'
    add_project_arguments(['-Mnostack_arrays', '-Mrecursive'], language : 'fortran')
  elif fortran_compiler.get_id() == 'nagfor'
    add_project_arguments('-recursive', language : 'fortran')  # What about stack/heap?
  endif
endif


# Set additional linker flags
# Zaikun 20240217: Fix https://github.com/libprima/prima/issues/158, which is due to the new linker
# implemented in Xcode 15 on macOS. It happens only if the Fortran compiler is ifort.
# An alternative is `add_link_options("-ld_classic")`, which forces Xcode to use the old linker.
# Will CMake adapt itself to the new linker later? Will a newer version of CMake make the fix unnecessary?
# See
# https://developer.apple.com/documentation/xcode-release-notes/xcode-15-release-notes#Linking
# https://stackoverflow.com/questions/77525544/apple-linker-warning-ld-warning-undefined-error-is-deprecated
# https://medium.com/@hackingwithcode/cmake-and-xcode-15-solving-the-undefined-error-puzzle-3c847e6d1008
if (build_machine.system() == 'darwin') and (fortran_compiler.get_id() == 'intel')
    add_project_link_arguments('-Wl,-undefined,dynamic_lookup', language : 'fortran')
endif


if build_machine.system() == 'windows'
    cmdfile = fs.read('cmdfile.gdb')
    cmdfile = cmdfile.replace('\n', '\r\n')
    configure_file(input: 'cmdfile.gdb',
        output: 'cmdfile.gdb',
        capture: true,
        command: [
            'echo',
            cmdfile
        ])
    # Since Windows does not have a concept of rpath, we put all the dlls and
    # executables in the same directory so the executables can find the dlls at
    # runtime.
    runtime_output_directory = meson.current_build_dir() / 'bin'
    runtime_output_directory_folder = custom_target(
        'runtime_output_directory_folder',
        input: 'meson.build',  # Can be any (existing) file, not actually used
        command: ['mkdir', '-p', runtime_output_directory],
        output: '.workaround'  # Can be any name, not actually used
        )
endif
subdir('fortran')

if get_option('prima_enable_c')
    add_languages('c')
    subdir('c')
endif

alias_target('examples', [fortran_examples, c_examples])


# Get version from git or the archive file
git = find_program('git', required : false)
if git.found()
    prima_version = run_command(git, 'describe', '--tags', '--always', '--dirty',
        check : true,
        capture : true).stdout().strip()
else
    # This may happen in the case of building for MacPorts. In this situation the
    # repo was presumably archived, in which case the contents of the .git-archival.txt file
    # should have been replace with the version. We read the file and see if it contains the
    # word 'describe', which would indicate that its contents were'nt replaced and this isn't
    # an archive. 
    prima_version = fs.read('.git-archival.txt')
    if prima_version.contains('describe')
        message('git was not detected and .git-archival.txt does not contain a version number.')
        prima_version = 'unknown'
    endif
endif

# Remove the leading v from PRIMA_VERSION, if it contains one.
if prima_version.startswith('v')
    prima_version = prima_version.substring(1)
endif
message('Setting PRIMA_VERSION to ' + prima_version)


# TODO: Figure out package config file
# TODO: Test in CI with all the different compilers and platforms
