project('prima', 'fortran')

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

configure_file(
    input: 'cmdfile.gdb',
    copy: true,
    output: 'cmdfile.gdb',  # This will install it into the build directory
    # TODO: I'm not sure if that's the right place to put it, but we'll deal with that later.
    # I wasn't able to get it to construct the file at build time. It's possible, but it gets
    # really complicated with all sort of shell redirection, particular since we use $ in
    # the file. Much more straightforward to just keep it around and copy it.
)

# custom_target

subdir('fortran')

if get_option('prima_enable_c')
    add_languages('c')
    subdir('c')
endif

alias_target('examples', fortran_examples)
