
prima_f_sources = [
  'common/linalg.f90',
  'common/pintrf.f90',
  'common/string.f90',
  'common/evaluate.f90',
  'common/preproc.f90',
  'common/univar.f90',
  'common/powalg.f90',
  'common/history.f90',
  'common/xinbd.f90',
  'common/message.f90',
  'common/fprint.f90',
  'common/ratio.f90',
  'common/debug.F90',
  'common/consts.F90',
  'common/checkexit.f90',
  'common/inf.F90',
  'common/redrho.f90',
  'common/memory.F90',
  'common/selectx.f90',
  'common/huge.F90',
  'common/infnan.F90',
  'common/shiftbase.f90',
  'common/infos.f90',
  'cobyla/cobyla.f90',
  'cobyla/cobylb.f90',
  'cobyla/geometry.f90',
  'cobyla/initialize.f90',
  'cobyla/trustregion.f90',
  'cobyla/update.f90',
  'bobyqa/bobyqa.f90',
  'bobyqa/bobyqb.f90',
  'bobyqa/geometry.f90',
  'bobyqa/initialize.f90',
  'bobyqa/trustregion.f90',
  'bobyqa/rescue.f90',
  'bobyqa/update.f90',
  'lincoa/update.f90',
  'lincoa/initialize.f90',
  'lincoa/getact.f90',
  'lincoa/trustregion.f90',
  'lincoa/geometry.f90',
  'lincoa/lincob.f90',
  'lincoa/lincoa.f90',
  'newuoa/initialize.f90',
  'newuoa/trustregion.f90',
  'newuoa/geometry.f90',
  'newuoa/update.f90',
  'newuoa/newuob.f90',
  'newuoa/newuoa.f90',
  'uobyqa/initialize.f90',
  'uobyqa/update.f90',
  'uobyqa/geometry.f90',
  'uobyqa/trustregion.f90',
  'uobyqa/uobyqb.f90',
  'uobyqa/uobyqa.f90',
]

# def_file = ''
# if build_machine.system() == 'windows' and get_option('default_library') == 'shared'
#   if fortran_compiler.get_id() == 'gcc'
#     def_file = 'primaf-GNU.def'
#   else
#     message('Assuming Intel symbol mangling')
#     def_file = 'primaf-Intel.def'
#   endif
# endif


defines = [
    '-DPRIMA_REAL_PRECISION=64',
    '-DPRIMA_INTEGER_KIND=0',
]
if build_machine.system() == 'windows' and meson.get_compiler('fortran').get_id().startswith('intel')
    # This was necessary so that #include's and other preprocessing directives in Fortran files
    # could be processed by the intel compiler. It seems that this should probably go into meson itself,
    # since CMake was able to add this on its own and meson adds this for some Fortran compilers but
    # not all.
    defines += ['-fpp']
endif

primaf = library('primaf',
  prima_f_sources,
  fortran_args : defines,
  install : true,
  pic : true,
  vs_module_defs: fortran_compiler.get_id() == 'gcc' ? 'primaf-GNU.def' : 'primaf-Intel.def',
)

if build_machine.system() == 'windows'
    configure_file(input: primaf.full_path(),
        output: runtime_output_directory / fs.name(primaf.full_path()),
        copy: true)
endif
# TODO: Add the examples to this file
# Make sure to add build_by_default : get_option('prima_enable_examples')
# And to add the tests under a similar flag

subdir('examples')