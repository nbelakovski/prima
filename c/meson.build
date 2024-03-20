c_standard = 'c_std=gnu99'

sources = ['cintrf.f90',
    'cobyla_c.f90',
    'lincoa_c.f90',
    'bobyqa_c.f90',
    'newuoa_c.f90',
    'uobyqa_c.f90',
    'prima.c'
]

# Export symbols on Windows. See more comments in fortran/CMakeLists.txt.
if build_machine.system() == 'windows' and get_option('default_library') == 'shared'
    sources += ['primac.def']
endif

primac = library('primac', sources: sources,
    c_static_args : '-DPRIMAC_STATIC',
    include_directories: include_directories('include'),
    link_with: primaf,
    pic: true,
    link_language: 'fortran',
    override_options : [c_standard])

if build_machine.system() == 'windows'
    # TODO: Let's see if this is really necessary
    # set_target_properties(primac PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif

# if (NOT BUILD_SHARED_LIBS)
#   target_compile_definitions(primac PUBLIC PRIMAC_STATIC) <- taken care of by c_static_args = '-DPRIMAC_STATIC'
#   target_link_libraries (primac INTERFACE ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}) <- not sure how to implement this one
#   # I believe the above command might not be necessary
# endif ()


common = declare_dependency(
    include_directories : include_directories('include'),
    link_with : primac)

examples = {
    'bobyqa_example_c': 'examples/bobyqa/bobyqa_example.c',
    'cobyla_example_c': 'examples/cobyla/cobyla_example.c',
    'lincoa_example_c': 'examples/lincoa/lincoa_example.c',
    'newuoa_example_c': 'examples/newuoa/newuoa_example.c',
    'uobyqa_example_c': 'examples/uobyqa/uobyqa_example.c'
}

example_executables = []
foreach name, source : examples
    example_exe = executable(name, sources : source, dependencies: common)
    test(name, example_exe, timeout: 10, suite: ['c_tests', name])
    example_executables += [example_exe]
endforeach

c_examples = alias_target('c_examples', example_executables)


# Gdb tests
gdb = find_program('gdb', required: false)
if not gdb.found()
    # On some macOS systems there is an extra 'g' in front
    gdb = find_program('ggdb', required: false)
endif

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


# TODO: There is a tests subdirectory we need to implement