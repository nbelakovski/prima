c_standard = 'c_std=gnu99,c99'

sources = ['cintrf.f90',
    'cobyla_c.f90',
    'lincoa_c.f90',
    'bobyqa_c.f90',
    'newuoa_c.f90',
    'uobyqa_c.f90',
    'prima.c'
]

defines = [
    '-DPRIMA_REAL_PRECISION=64',
    '-DPRIMA_INTEGER_KIND=0',
]
if build_machine.system() == 'windows' and get_option('default_library') == 'shared'
    # It seems CMake handles this automatically but Meson does not
    defines += ['-Dprimac_EXPORTS']
endif

primac = library('primac', sources: sources,
    c_args: defines,
    c_static_args : '-DPRIMAC_STATIC',
    include_directories: include_directories('include'),
    link_language: 'fortran',
    link_with: primaf,
    override_options : [c_standard],
    pic: true,
    vs_module_defs: 'primac.def',
)

if build_machine.system() == 'windows'
    # TODO: Let's see if this is really necessary
    # set_target_properties(primac PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif

# if (NOT BUILD_SHARED_LIBS)
#   target_compile_definitions(primac PUBLIC PRIMAC_STATIC) <- taken care of by c_static_args = '-DPRIMAC_STATIC'
#   target_link_libraries (primac INTERFACE ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) <- not sure how to implement this one
#   # I believe the above command might not be necessary
# endif ()

if (meson.get_compiler('c').get_id() == 'clang') and (meson.get_compiler('fortran').get_id() == 'flang')
    # With flang, meson insists upon linking -lflang and -lpgmath. Our program links successfully without those,
    # (CMake does not include them and succeeds) but meson insists. However meson does not know where to find these
    # libraries, so we need to supply the path ourselves. The code below finds the path of flang, goes back a directory,
    # and adds the 'lib/' directory. It is not designed to be cross platform since we do not test this compiler on
    # macOS or Windows.
    flang_location = run_command('which', 'flang', capture: true, check: true).stdout().strip()
    # Remove the last 9 characters ('bin/flang') and add lib/
    flang_libs = flang_location.substring(0, -9) / 'lib'
    message('flang_libs: -L' + flang_libs)
    common = declare_dependency(
        compile_args: defines,
        include_directories : include_directories('include'),
        link_args : ['-L' + flang_libs],
        link_with : [primac, primaf])
else
    common = declare_dependency(
        compile_args: defines,
        include_directories : include_directories('include'),
        link_with : [primac, primaf])
endif

examples = {
    'bobyqa_example_c': 'examples/bobyqa/bobyqa_example.c',
    'cobyla_example_c': 'examples/cobyla/cobyla_example.c',
    'lincoa_example_c': 'examples/lincoa/lincoa_example.c',
    'newuoa_example_c': 'examples/newuoa/newuoa_example.c',
    'uobyqa_example_c': 'examples/uobyqa/uobyqa_example.c'
}

example_executables = []
foreach name, source : examples
    example_exe = executable(name, sources : source, dependencies: common,
        override_options: [c_standard])
    test(name, example_exe, timeout: 10, suite: ['c_tests', name])
    example_executables += [example_exe]
endforeach

c_examples = alias_target('c_examples', example_executables)


# Within CI, we'd like to run with gdb so that if there's a segfault the logs will have a stacktrace we can use to investigate.
if get_option('CI') and build_machine.system() != 'darwin'  # Apple security policy will not allow running gdb in CI
    gdb = find_program('gdb', required: false)

    if gdb.found()
        foreach example_executable : example_executables
            test(example_executable.name() + '_gdb', gdb,
                args : ['-batch', '--command=' + meson.project_source_root() / 'cmdfile.gdb', example_executable.full_path()],
                # On my machine I ran into some intermittent failures that were resolved by ensuring these tests do not
                # run in parallel
                is_parallel: false,
                suite: ['gdb', example_executable.name() + '_gdb'],
                timeout: 10,
            )
        endforeach
    else
        message('gdb not found, not running gdb tests')
    endif
endif


# TODO: There is a tests subdirectory we need to implement