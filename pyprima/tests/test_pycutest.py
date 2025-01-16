import pytest
# This exists mainly for those CI tests in which cutest/pycutest/optiprofiler are not installed.
optiprofiler = pytest.importorskip("optiprofiler", exc_type=ImportError)

from optiprofiler.problems import load_cutest_problem
from pyprima import minimize, Bounds, LinearConstraint, NonlinearConstraint
import numpy as np
import sys


'''
This module tests various problem from the CUTEST set in order to really stress test
the implementation and also cover some cases not covered by the naive tests in
test_end_to_end.py. The list is semi-arbitrary, some of these helped to fine bugs
when testing the Python implementation against the Fortran one.
'''


@pytest.fixture(autouse=True, scope='module')
def set_comparing():
    # This is a hack to force these tests to use manual math instead of optimized
    # numpy or other routines. This should ensure we have the same results across
    # different architectures.
    sys.modules['pyprima'].common.linalg.COMPARING = True
    yield
    sys.modules['pyprima'].common.linalg.COMPARING = False


def get_constraints(problem):
    constraints = []
    if problem.m_linear_ub > 0:
        constraints.append(LinearConstraint(problem.a_ub, -np.inf, problem.b_ub))
    if problem.m_linear_eq > 0:
        constraints.append(LinearConstraint(problem.a_eq, problem.b_eq, problem.b_eq))
    if problem.m_nonlinear_ub > 0:
        constraints.append(NonlinearConstraint(problem.c_ub, -np.inf, np.zeros(problem.m_nonlinear_ub)))
    if problem.m_nonlinear_eq > 0:
        constraints.append(NonlinearConstraint(problem.c_eq, np.zeros(problem.m_nonlinear_eq), np.zeros(problem.m_nonlinear_eq)))
    return constraints


def run_problem(name, expected_x, expected_f, expected_constraints, expected_nf):
    problem = load_cutest_problem(name)
    constraints = get_constraints(problem)
    bounds = Bounds(problem.lb, problem.ub)
    result = minimize(problem.fun, problem.x0, method='cobyla', constraints=constraints, bounds=bounds)
    assert np.allclose(result.x, expected_x, atol=1e-15)
    assert np.isclose(result.f, expected_f, atol=1e-15)
    assert np.allclose(result.constr, expected_constraints, atol=1e-15)
    assert result.nf == expected_nf


@pytest.mark.order(2)  # This test takes the second longest
def test_errinbar():
    # Expected values are just obtained from running the problem and collecting the results
    # If future changes improve the algorithm, these values may need to be updated.
    expected_x = np.array([-245.5081612710879710448,  -54.2958188399598853380,
                            262.7230936253279196535,  -45.9309872501519365073,
                             -2.2691358288677578869,    7.6733425543485722642,
                            -31.0344774501767943775,   26.9394969852641779084,
                             -1.1718811167782969829,   -2.8508188399598868301,
                             -2.8508188399598868301,   -2.8508188399598868301,
                             62.5293726076140217174,    1.0329450380513451879,
                              7.1143643043640603096,    3.3150533074468162553,
                              1.1289565001348098594,   13.0411964959208557246])
    expected_f = 203.48674841371607
    expected_constraints = np.array([
          3.4958188399598881801,  -4.8690127498480606505, -58.4733425543485694220,
        -77.7394969852641679608,   1.8168811167782970006,   3.4958188399598868479,
          3.4958188399598868479,   3.4958188399598868479, -61.8843726076140185910,
         -0.3879450380513451702,  -6.4693643043640598478,  -2.6700533074468162376,
         -0.4839565001348098416, -12.3961964959208561510, -31.2248315898079482622,
          3.4957795914671430104,  -3.4963193946205968210,   3.4957779371938499935,
          3.4955724606965077328,  -3.4957078923282551841,  -3.4968691556686337663,
         -3.4959635816089189575,   3.4948114345599776698,  -3.4957795914671430104,
          3.4963193946205968210,  -3.4957779371938499935,  -3.4955724606965077328,
          3.4957078923282551841,   3.4968691556686337663,   3.4959635816089189575,
         -3.4948114345599776698])
    expected_nf = 9000
    run_problem('ERRINBAR', expected_x, expected_f, expected_constraints, expected_nf)


def test_palmer2c():
    expected_x = np.array([0.8853310838319223830, 0.8885299788096961970, 0.8805439805147466936,
        0.7783464729241426072, 0.5850067768175214455, 0.0316686276050338403,
        0.0244429707945372325, -0.0163015984157743772])
    expected_f = 1414.0819761003331
    expected_constraints = np.array([])
    expected_nf = 4000
    run_problem('PALMER2C', expected_x, expected_f, expected_constraints, expected_nf)


def test_palmer3b():
    expected_x = np.array([1.2090027787973145479, 7.5461811633941122679, 0.4433420649209307562,
        0.0387612788701699879])
    expected_f = 324.6120596974852
    expected_constraints = np.array([-0.4433320649209307462, -0.0387512788701699848])
    expected_nf = 2000
    run_problem('PALMER3B', expected_x, expected_f, expected_constraints, expected_nf)


def test_tfi3():
    expected_x = np.array([1.0052196032967017914, -0.1118710807081673281, -0.3933485225885347547])
    expected_f = 4.301460324843021
    expected_constraints = np.array([
        -5.2196032967017913506e-03, -4.1615476373613180527e-03,
        -3.2246822735029212481e-03, -2.4086482051270952098e-03,
        -1.7128464322335723580e-03, -1.1364439548220417464e-03,
        -6.7836977289303934668e-04, -3.3732688644627639718e-04,
        -1.1178629548169727315e-04,  4.4408920985006261617e-16,
         2.2204460492503130808e-16, -1.0960729548215031315e-04,
        -3.2643888644678931144e-04, -6.4790977289375106807e-04,
        -1.0712459548231079154e-03, -1.5934894322346959683e-03,
        -2.2115062051284439804e-03, -2.9219972735047283763e-03,
        -3.7215016373631781832e-03, -4.6064132967039705946e-03,
        -5.5729842515268579461e-03, -6.6173405018321584947e-03,
        -7.7354830476199509803e-03, -8.9233058888898986183e-03,
        -1.0176604025642022044e-02, -1.1491079457876418601e-02,
        -1.2862357185593364939e-02, -1.4285991208792370166e-02,
        -1.5757473527473808694e-02, -1.7272248141637414065e-02,
        -1.8825719051283495986e-02, -2.0413257256411743157e-02,
        -2.2030212757022216152e-02, -2.3671921553115149450e-02,
        -2.5333716644690307263e-02, -2.7010934031747702022e-02,
        -2.8698923714287394304e-02, -3.0393054692309462439e-02,
        -3.2088722965813776256e-02, -3.3781361534800380397e-02,
        -3.5466441399269243995e-02, -3.7139485559220530853e-02,
        -3.8796067014653989879e-02, -4.0431820765569725928e-02,
        -4.2042444811967838447e-02, -4.3623709153848166942e-02,
        -4.5171455791210846087e-02, -4.6681605724055774687e-02,
        -4.8150161952382974917e-02, -4.9573211476192646785e-02,
        -5.0946932295484415043e-02, -5.2267591410258495976e-02,
        -5.3531548820514918141e-02, -5.4735260526253770585e-02,
        -5.5875281527474673915e-02, -5.6948261824178003643e-02,
        -5.7950953416363470971e-02, -5.8880207304031539906e-02,
        -5.9732975487181705354e-02, -6.0506310965814047442e-02,
        -6.1197368739928981185e-02, -6.1803404809525952501e-02,
        -6.2321776174605214571e-02, -6.2749936835166941762e-02,
        -6.3085444791210809612e-02, -6.3325952042737054803e-02,
        -6.3469211589745611057e-02, -6.3513071432236389846e-02,
        -6.3455471570209587284e-02, -6.3294449003664987607e-02,
        -6.3028130732602605235e-02, -6.2654733757022573748e-02,
        -6.2172564076924952126e-02, -6.1580011692309422067e-02,
        -6.0875553603176313366e-02, -6.0057748809525413058e-02,
        -5.9125236311357021179e-02, -5.8076733108670675065e-02,
        -5.6911032201466649205e-02, -5.5627003589744927758e-02,
        -5.4223586273505763877e-02, -5.2699791252748595660e-02,
        -5.1054694527473909460e-02, -4.9287441097681239377e-02,
        -4.7397237963371252611e-02, -4.5383353124543224233e-02,
        -4.3245114581197507952e-02, -4.0981907333334288701e-02,
        -3.8593172380953277845e-02, -3.6078400724054549009e-02,
        -3.3437139362637924300e-02, -3.0668980296703729493e-02,
        -2.7773565526252097513e-02, -2.4750580051282500271e-02,
        -2.1599752871795274700e-02, -1.8320855987790318764e-02,
        -1.4913700399267648677e-02, -1.1378134106227122402e-02,
        -7.7140431086689664752e-03, -3.9213474065933340285e-03,
         2.7755575615628913511e-16 ])
    expected_nf = 15
    run_problem('TFI3', expected_x, expected_f, expected_constraints, expected_nf)


def test_hs103():
    # This one hits the section in trustregion.py which scales the problem if A
    # contains large values.
    expected_x = np.array([2.5289807396883552393, 0.4636478250644894272, 3.4406699575922705669,
        9.1465471863584326684, 2.2192889216323616886, 2.8022288337905920663,
        0.0173213436307221025])
    expected_f = 3000.2102326770078
    expected_constraints = np.array([
        -2.4289807396883551505e+00, -3.6364782506448944943e-01,
        -3.3406699575922704781e+00, -9.0465471863584330237e+00,
        -2.1192889216323615997e+00, -2.7022288337905919775e+00,
        -7.3213436307221022720e-03, -7.4710192603116443166e+00,
        -9.5363521749355104618e+00, -6.5593300424077298771e+00,
        -8.5345281364156733162e-01, -7.7807110783676378674e+00,
        -7.1977711662094083778e+00, -9.9826786563692770926e+00,
        -4.8208275891177221339e-01,  2.1023382818340427547e-01,
         2.1023382797198550409e-01, -7.4925408486881206471e-01,
        -2.9002102326770077525e+03,  2.1023267700795286728e-01])
    expected_nf = 3500
    run_problem('HS103', expected_x, expected_f, expected_constraints, expected_nf)


def test_cresc4():
    expected_x = np.array([
        -2.4712333082037314824e+01,  1.2118172433432175539e-01,
         1.0544855650501498978e+00,  2.3019400598944947944e+01,
        -2.3698388504182490846e-18,  3.9000000000000001332e-01])
    expected_f = 2.2014732007195974
    expected_constraints = np.array([
        -1.0544855550501499586e+00, -2.2019400598944947944e+01,
         2.3698388504182490846e-18,  0.0000000000000000000e+00,
        -6.2831852000000001368e+00, -1.9744556915237687633e-03,
        -5.2844327442343001167e+01, -1.1217527730463550117e+00,
        -3.1772978295997518217e+00, -6.3702587570906854531e-01,
        -3.6620247269370338472e+00, -1.1906818900346176626e+00,
        -2.7381994360305725422e+01])
    expected_nf = 3000
    run_problem('CRESC4', expected_x, expected_f, expected_constraints, expected_nf)


def test_mgh10ls():
    # This one also hits the section in trustregion.py which scales the problem if A
    # contains large values.
    expected_x = np.array([1.4950790839504562481e-03, 3.9999949550186062697e+05,
        2.5001006103891704697e+04])
    expected_f = 1366860355.936367
    expected_constraints = np.array([])
    expected_nf = 51
    run_problem('MGH10LS', expected_x, expected_f, expected_constraints, expected_nf)


@pytest.mark.order(1)  # This test takes the longest
def test_tenbars1():
    expected_x = np.array([
         1.9568934516948072542e+03,  3.3869509993142941084e+02,
         5.1084410783004557288e+02,  7.1451015340321816893e+02,
         6.7889550790383304957e+02,  5.5925898284379888992e+02,
         9.4664217112688754696e+02,  5.6527088845435753228e+02,
         8.5148703700913730419e-01, -1.3262297582844631005e+00,
        -1.3262297582844631005e+00, -1.3262297582844631005e+00,
        -1.3262297582844631005e+00, -6.2387954220169796263e-03,
        -1.3262297582844631005e+00, -1.3262297582844631005e+00,
        -1.3262297582844631005e+00, -1.3262297582844631005e+00])
    expected_f = -30.27643654696821
    expected_constraints = np.array([
        -3.8949509993142942221e+02, -7.6531015340321812346e+02,
        -6.1005898284379884444e+02, -6.1607088845435748681e+02,
        -2.0648703700913728643e-01,  1.9712297582844631183e+00,
         1.9712297582844631183e+00,  1.9712297582844631183e+00,
         1.9712297582844631183e+00,  6.5123879542201701387e-01,
         1.9712297582844631183e+00,  1.9712297582844631183e+00,
         1.9712297582844631183e+00,  1.9712297582844631183e+00,
        -3.9867505347178877173e+02,  1.9711816402637432066e+00,
         1.9712297463431205369e+00,  1.9712297462376682233e+00,
         1.9712419862444789942e+00,  1.9712293413148502808e+00,
         1.9712297449648037428e+00,  1.9712297458175953579e+00,
         1.9712419824544298308e+00, -1.9711816402637432066e+00,
        -1.9712297463431205369e+00, -1.9712297462376682233e+00,
        -1.9712419862444789942e+00, -1.9712293413148502808e+00,
        -1.9712297449648037428e+00, -1.9712297458175953579e+00,
        -1.9712419824544298308e+00])
    expected_nf = 9000
    run_problem('TENBARS1', expected_x, expected_f, expected_constraints, expected_nf)


def test_biggs3():
    expected_x = np.array([0.9999739462811082502, 9.9987085888009339385, 4.9992405910694346360])
    expected_f = 1.0631133951128183e-08
    expected_constraints = np.array([])
    expected_nf = 1500
    run_problem('BIGGS3', expected_x, expected_f, expected_constraints, expected_nf)


def test_biggs6():
    expected_x = np.array([1.2194245092803301933, 8.5197890909932567638, 1.1508892999691997527,
        3.8853547380197808181, 3.4765466485421625542, 2.4764750429775079787])
    expected_f = 0.009233751892945407
    expected_constraints = np.array([])
    expected_nf = 3000
    run_problem('BIGGS6', expected_x, expected_f, expected_constraints, expected_nf)


def test_degenlpb():
    expected_x = np.array([
         2.5048137222646543742e-01,  8.1652820831984397262e-04,
         2.8086877200736666549e-02,  9.9836359424439302668e-02,
         3.9729580203073201622e-06,  1.3602542024738497294e-04,
         4.8342827214562932199e-04,  5.6699356477458474901e-04,
         1.0247844713503989709e-03,  1.9602965525651544487e-01,
        -2.5390800574482325636e-13, -2.5390800733077673568e-13,
        -2.5390800458724593676e-13, -9.5814957530582223200e-14,
        -2.5353478905985151653e-13,  1.9962073754867938440e-03,
        -2.5390811415199054935e-13,  1.6586456462450486181e-06,
         3.9275644489467232551e-03,  1.2011160395190988333e-03])
    expected_f = -30.731246817983664
    expected_constraints = np.array([
        -2.5048137222646543742e-01, -8.1652820831984397262e-04,
        -2.8086877200736666549e-02, -9.9836359424439302668e-02,
        -3.9729580203073201622e-06, -1.3602542024738497294e-04,
        -4.8342827214562932199e-04, -5.6699356477458474901e-04,
        -1.0247844713503989709e-03, -1.9602965525651544487e-01,
         2.5390800574482325636e-13,  2.5390800733077673568e-13,
         2.5390800458724593676e-13,  9.5814957530582223200e-14,
         2.5353478905985151653e-13, -1.9962073754867938440e-03,
         2.5390811415199054935e-13, -1.6586456462450486181e-06,
        -3.9275644489467232551e-03, -1.2011160395190988333e-03,
        -7.4951862777353461809e-01, -9.9918347179168021110e-01,
        -9.7191312279926334039e-01, -9.0016364057556064182e-01,
        -9.9999602704197965153e-01, -9.9986397457975262348e-01,
        -9.9951657172785435268e-01, -9.9943300643522536841e-01,
        -9.9897521552864965155e-01, -8.0397034474348449962e-01,
        -1.0000000000002540190e+00, -1.0000000000002540190e+00,
        -1.0000000000002537970e+00, -1.0000000000000959233e+00,
        -1.0000000000002535749e+00, -9.9800379262451321960e-01,
        -1.0000000000002540190e+00, -9.9999834135435372584e-01,
        -9.9607243555105329236e-01, -9.9879888396048088772e-01,
        -2.5413005033669833210e-13, -2.8724245204614362592e-13,
        -2.4270863097086703419e-13, -2.5389065849701353272e-13,
        -2.7571001037784981236e-13, -2.5407898441437426484e-13,
        -2.5391320990220123122e-13, -2.4266699760744359082e-13,
         2.5390952361481478050e-13, -2.5402759323139845193e-13,
         2.5390774823375733549e-13,  2.5390795469244733799e-13,
        -2.5390729730980910473e-13, -2.5379698342931078514e-13,
         2.5391494462567720802e-13,  2.5413005033669833210e-13,
         2.8724245204614362592e-13,  2.4270863097086703419e-13,
         2.5389065849701353272e-13,  2.7571001037784981236e-13,
         2.5407898441437426484e-13,  2.5391320990220123122e-13,
         2.4266699760744359082e-13, -2.5390952361481478050e-13,
         2.5402759323139845193e-13, -2.5390774823375733549e-13,
        -2.5390795469244733799e-13,  2.5390729730980910473e-13,
         2.5379698342931078514e-13, -2.5391494462567720802e-13])
    expected_nf = 97
    run_problem('DEGENLPB', expected_x, expected_f, expected_constraints, expected_nf)


def test_hs102():
    # This is a very important test, because after nearly all of the math was switched
    # to manual math, 99% of problems were identical except this one, and it was because
    # of a non-math related bug in the code. It was a rather serious bug and it's surprising
    # that it wasn't caught earlier, but it's a good reminder that these things happen, and
    # also why it's important to check the math with precision algorithms instead of just
    # ignoring issues when the differences appear close to machine epsilon.
    expected_x = np.array([2.4754575633775468546, 0.5541250457494729664, 3.6444439553239513785,
        8.1630851993117925502, 1.5198804708746382897, 1.9600795356050697560,
        0.0196160573415092507])
    expected_f = 3000.076360741547
    expected_constraints = np.array([
        -2.3754575633775467658e+00, -4.5412504574947298863e-01,
        -3.5444439553239512897e+00, -8.0630851993117929055e+00,
        -1.4198804708746382008e+00, -1.8600795356050696672e+00,
        -9.6160573415092504695e-03, -7.5245424366224531454e+00,
        -9.4458749542505273666e+00, -6.3555560446760486215e+00,
        -1.8369148006882074498e+00, -8.4801195291253623765e+00,
        -8.0399204643949300220e+00, -9.9803839426584914918e+00,
        -6.0742857186875132136e-01,  7.6362201567565679561e-02,
         7.6362201333172202711e-02, -6.3020233552259341536e-01,
        -2.9000763607415465231e+03,  7.6360741546495691789e-02,
    ])
    expected_nf = 3500
    run_problem('HS102', expected_x, expected_f, expected_constraints, expected_nf)
