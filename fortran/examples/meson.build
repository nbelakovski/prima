bobyqa_example_1 = executable('bobyqa_example_1', 'bobyqa/bobyqa_example_1.f90', link_with: primaf)
bobyqa_example_2 = executable('bobyqa_example_2', 'bobyqa/bobyqa_example_2.f90', link_with: primaf)
cobyla_example_1 = executable('cobyla_example_1', 'cobyla/cobyla_example_1.f90', link_with: primaf)
cobyla_example_2 = executable('cobyla_example_2', 'cobyla/cobyla_example_2.f90', link_with: primaf)
lincoa_example_1 = executable('lincoa_example_1', 'lincoa/lincoa_example_1.f90', link_with: primaf)
lincoa_example_2 = executable('lincoa_example_2', 'lincoa/lincoa_example_2.f90', link_with: primaf)
newuoa_example_1 = executable('newuoa_example_1', 'newuoa/newuoa_example_1.f90', link_with: primaf)
newuoa_example_2 = executable('newuoa_example_2', 'newuoa/newuoa_example_2.f90', link_with: primaf)
uobyqa_example_1 = executable('uobyqa_example_1', 'uobyqa/uobyqa_example_1.f90', link_with: primaf)
uobyqa_example_2 = executable('uobyqa_example_2', 'uobyqa/uobyqa_example_2.f90', link_with: primaf)

fortran_examples = alias_target('fortran_examples', [bobyqa_example_1,
    bobyqa_example_2,
    cobyla_example_1,
    cobyla_example_2,
    lincoa_example_1,
    lincoa_example_2,
    newuoa_example_1,
    newuoa_example_2,
    uobyqa_example_1,
    uobyqa_example_2]
)
