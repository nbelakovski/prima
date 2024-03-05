// project includes
#include <prima/prima.h>
// 3rd party includes
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// standard includes
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

// The following class is necessary to ensure that the static variables used to
// store the objective/constraint/callback functions are automatically reset
// upon normal or abnormal exit from the _minimize function. If we don't do this,
// we can get various errors in Python, from hanging, to warnings about the GIL, to
// segfaults. Simply setting the static variables to py::none() is sufficient.
class SelfCleaningPyObject {
  py::object &obj;
  public:
    SelfCleaningPyObject(py::object &obj) : obj(obj) {}
    ~SelfCleaningPyObject() {
      obj = py::none();
    }
};

struct PRIMAResult {
    // Construct PRIMAResult from prima_result_t
    PRIMAResult(const prima_result_t& result, const int num_vars, const int num_constraints)  :
    x(num_vars, result.x),
    success(prima_is_success(result)),
    status(result.status),
    message(result.message),
    fun(result.f),
    nfev(result.nf),
    maxcv(result.cstrv),
    nlconstr(num_constraints, result.nlconstr) {}

    std::string repr() const {
      std::string repr = "PRIMAResult(";
      repr = repr + 
        "x=" + std::string(pybind11::repr(x)) + ", " +
        "success=" + std::to_string(success) + ", " +
        "status=" + std::to_string(status) + ", " +
        "message=" + "\'" + message + "\'" + ", " +
        "fun=" + std::to_string(fun) + ", " +
        "nfev=" + std::to_string(nfev) + ", " +
        "maxcv=" + std::to_string(maxcv) + ", " +
        "nlconstr=" + std::string(pybind11::repr(nlconstr)) + 
        ")";
      return repr;
    }

    pybind11::array_t<double, pybind11::array::c_style> x;        // final point
    bool success;                                                 // whether the solver terminated with an error or not
    int status;                                                   // exit code
    std::string message;                                          // error message
    double fun;                                                   // objective value
    int nfev;                                                     // number of objective function calls
    double maxcv;                                                 // constraint violation (cobyla & lincoa)
    pybind11::array_t<double, pybind11::array::c_style> nlconstr; // non-linear constraint values, of size m_nlcon (cobyla only)
};


prima_algorithm_t pystr_method_to_algorithm(const pybind11::str& method_) {
  const std::string method(method_);
  if (method == "bobyqa") { return PRIMA_BOBYQA; }
  else if (method == "cobyla") { return PRIMA_COBYLA; }
  else if (method == "lincoa") { return PRIMA_LINCOA; }
  else if (method == "newuoa") { return PRIMA_NEWUOA; }
  else if (method == "uobyqa") { return PRIMA_UOBYQA; }
  else { throw std::invalid_argument("method must be one of BOBYQA, COBYLA, LINCOA, NEWUOA, or UOBYQA"); }
}

PYBIND11_MODULE(_prima, m) {
#ifdef VERSION_INFO
    #define STRINGIFY(x) #x
    #define MACRO_STRINGIFY(x) STRINGIFY(x)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<PRIMAResult>(m, "PRIMAResult")
      .def_readwrite("x", &PRIMAResult::x)
      .def_readwrite("success", &PRIMAResult::success)
      .def_readwrite("status", &PRIMAResult::status)
      .def_readwrite("message", &PRIMAResult::message)
      .def_readwrite("fun", &PRIMAResult::fun)
      .def_readwrite("nfev", &PRIMAResult::nfev)
      .def_readwrite("maxcv", &PRIMAResult::maxcv)
      .def_readwrite("nlconstr", &PRIMAResult::nlconstr)
      .def("__repr__", &PRIMAResult::repr);

    py::enum_<prima_message_t>(m, "PRIMAMessage")
      .value("NONE", PRIMA_MSG_NONE)
      .value("EXIT", PRIMA_MSG_EXIT)
      .value("RHO", PRIMA_MSG_RHO)
      .value("FEVL", PRIMA_MSG_FEVL)
      .export_values();


    m.def("minimize", [](const py::function& python_objective_function,
                      py::array_t<double, py::array::c_style>& py_x0,
                      const py::tuple& args,

                      // All arguments which accept None as a default value must be
                      // of type py::object as opposed to their natural type.
                      // https://github.com/pybind/pybind11/issues/4956
                      const py::object& method,
                      const py::object& lb,
                      const py::object& ub,
                      const py::object& A_eq,
                      const py::object& b_eq,
                      const py::object& A_ineq,
                      const py::object& b_ineq,
                      const py::object& constraint_function,
                      const py::object& python_callback_function,
                      const py::object& options_dict)
    {
      // Reference for how to go from a python function to a C style function pointer: https://stackoverflow.com/questions/74480093
      // Basically, we need to create a function with a C signature that calls the python function, and
      // so the python function needs to exist in a scope outside that function with the C signature but still
      // be callable from within that function. Hence why we need to to be static and why we can't just capture
      // it within the lambda below (lambdas that capture variables cannot decay into function pointers).
      static py::function python_objective_function_holder;
      python_objective_function_holder = std::move(python_objective_function);
      auto cleaner_1 = SelfCleaningPyObject(python_objective_function_holder);
      static py::object python_callback_function_holder;
      python_callback_function_holder = std::move(python_callback_function);
      auto cleaner_2 = SelfCleaningPyObject(python_callback_function_holder);
      static py::object python_constraint_function_holder;
      python_constraint_function_holder = std::move(constraint_function);
      auto cleaner_3 = SelfCleaningPyObject(python_constraint_function_holder);
      // Storing the shape lets us handle both scalar inputs for x0 as well as 1D arrays
      static py::array::ShapeContainer x0_shape;  // no cleaner required for this one since it's just a wrapped std::vector
      if (py_x0.ndim() == 0) {
        x0_shape = {};
      } else if (py_x0.ndim() == 1) {
        x0_shape = {py_x0.shape(0)};
      } else {
        throw std::invalid_argument("x0 must be a scalar or a 1D array");
      }

      // Initialize the problem
      prima_problem_t problem;
      prima_init_problem(&problem, py_x0.size());
      problem.x0 = py_x0.mutable_data();

      // Process options. We do this first in order to see if the user provided the number of nonlinear constraints.
      prima_options_t options;
      prima_init_options(&options);
      if (options_dict.is_none() == false) {
        if(options_dict.contains("ftarget"))   { options.ftarget   = options_dict["ftarget"].cast<double>(); }
        if(options_dict.contains("iprint"))    { options.iprint    = options_dict["iprint"].cast<int>(); }
        if(options_dict.contains("maxfev"))    { options.maxfun    = options_dict["maxfev"].cast<int>(); }
        if(options_dict.contains("maxfun"))    { options.maxfun    = options_dict["maxfun"].cast<int>(); }
        if(options_dict.contains("npt"))       { options.npt       = options_dict["npt"].cast<int>(); }
        if(options_dict.contains("rhobeg"))    { options.rhobeg    = options_dict["rhobeg"].cast<double>(); }
        if(options_dict.contains("rhoend"))    { options.rhoend    = options_dict["rhoend"].cast<double>(); }
        // These go into problem, not options. The options dictionary is the only way to pass it in.
        if(options_dict.contains("m_nlcon"))   { problem.m_nlcon   = options_dict["m_nlcon"].cast<int>(); }
        if(options_dict.contains("f0"))        { problem.f0        = options_dict["f0"].cast<double>(); }
        if(options_dict.contains("nlconstr0")) {
          auto nlconstr0_buffer_info = options_dict["nlconstr0"].cast<py::buffer>().request();
          if (nlconstr0_buffer_info.format != "d")
          {
            throw std::invalid_argument("nlconstr0 must be a double array");
          }
          problem.nlconstr0 = (double*) nlconstr0_buffer_info.ptr;
        }
      }
      options.data = (void*)&args;

      prima_algorithm_t algorithm = pystr_method_to_algorithm(method);

      if ( algorithm == PRIMA_COBYLA ) {
        if (python_constraint_function_holder.is_none()) {
          throw std::invalid_argument("constraint_function must be provided if nonlinear constraints are provided");
        }

        problem.calcfc = [](const double x[], double *f, double constr[], const void *data) {
          // In order for xlist to not copy the data from x, we need to provide a dummy base object
          // Ideally pybind11 would provide a facility to do this instead of us having to do this hacky
          // thing but oh well. Reference: https://github.com/pybind/pybind11/issues/323#issuecomment-575717041
          py::none dummybaseobject;
          py::array_t<double> xlist(x0_shape, x, dummybaseobject);
          py::args args = *((py::args*)data);
          double result;
          py::object constraints;
          if (args.size() > 0) {
            result = python_objective_function_holder(xlist, args).cast<double>();
            constraints = python_constraint_function_holder(xlist, args);
          } else {
            result = python_objective_function_holder(xlist).cast<double>();
            constraints = python_constraint_function_holder(xlist);
          }

          *f = result;

          try
          {
            double constraint = constraints.cast<double>();
            *constr = constraint;
          }
          catch(const std::exception& e)
          {
            try
            {
              py::buffer_info constr_list_buffer_info = constraints.cast<py::buffer>().request();
              if (constr_list_buffer_info.format != "d")
              {
                throw std::invalid_argument("constraint_function must return a double array");
              }


              // We need to copy. We cannot set the pointer since we are not passed a pointer-to-pointer
              for (int i = 0; i < constr_list_buffer_info.size; i++) {
                constr[i] = ((double*)constr_list_buffer_info.ptr)[i];
              }
            }
            catch(const std::exception& e)
            {
              throw(std::invalid_argument("constraint_function must return a double or a double array"));
            }
            
          }
          
          
        };

      } else {
        // For all other algorithms we have a different, simpler signature for the objective function
        problem.calfun = [](const double x[], double *f, const void *data) {
          // In order for xlist to not copy the data from x, we need to provide a dummy base object
          // Ideally pybind11 would provide a facility to do this instead of us having to do this hacky
          // thing but oh well. Reference: https://github.com/pybind/pybind11/issues/323#issuecomment-575717041
          py::none dummybaseobject;
          py::array_t<double> xlist(x0_shape, x, dummybaseobject);
          py::args args = *((py::args*)data);
          double result;
          if (args.size() > 0) {
            result = python_objective_function_holder(xlist, args).cast<double>();
          } else {
            result = python_objective_function_holder(xlist).cast<double>();
          }
          *f = result;
        };
      }

      prima_callback_t cpp_callback_function_wrapper = nullptr;
      if (python_callback_function_holder.is_none() == false) {
        cpp_callback_function_wrapper = [](const int n, const double x[], const double f, int nf, int tr,
                                const double cstrv, int m_nlcon, const double nlconstr[], bool *terminate) {
          // In order for xlist to not copy the data from x, we need to provide a dummy base object
          // Ideally pybind11 would provide a facility to do this instead of us having to do this hacky
          // thing but oh well. Reference: https://github.com/pybind/pybind11/issues/323#issuecomment-575717041
          py::none dummybaseobject;
          py::array_t<double> xlist(n, x, dummybaseobject);
          py::array_t<double> nlconstrlist(m_nlcon, nlconstr, dummybaseobject);
          bool result = python_callback_function_holder(xlist, f, nf, tr, cstrv, nlconstrlist).cast<bool>();
          *terminate = result;
        };
      }
      options.callback = cpp_callback_function_wrapper;

      //=====================
      //    Handle Bounds
      //=====================
      if (lb.is_none() == false && ub.is_none() == false) {
        // Use the buffer protocol to avoid copying
        py::buffer_info lb_buffer_info  = lb.cast<py::buffer>().request();
        if (lb_buffer_info.format != "d")
        {
          throw std::invalid_argument("lb must be a double array");
        }
        problem.xl = (double*) lb_buffer_info.ptr;
        py::buffer_info ub_buffer_info  = ub.cast<py::buffer>().request();
        if (ub_buffer_info.format != "d")
        {
          throw std::invalid_argument("ub must be a double array");
        }
        problem.xu = (double*) ub_buffer_info.ptr;
      }

      //==============================
      //  Handle Linear Constraints
      //==============================
      if(A_eq.is_none() == false) {
        py::buffer_info A_eq_buffer_info  = A_eq.cast<py::buffer>().request();
        if (A_eq_buffer_info.format != "d")
        {
          throw std::invalid_argument("A_eq must be a double array");
        }
        problem.m_eq = A_eq_buffer_info.shape[0];
        problem.Aeq = (double*) A_eq_buffer_info.ptr;
        py::buffer_info b_eq_buffer_info  = b_eq.cast<py::buffer>().request();
        if (b_eq_buffer_info.format != "d")
        {
          throw std::invalid_argument("b_eq must be a double array");
        }
        problem.beq = (double*) b_eq_buffer_info.ptr;
      }
      if(A_ineq.is_none() == false) {
        py::buffer_info A_ineq_buffer_info  = A_ineq.cast<py::buffer>().request();
        if (A_ineq_buffer_info.format != "d")
        {
          throw std::invalid_argument("A_ineq must be a double array");
        }
        problem.m_ineq = A_ineq_buffer_info.shape[0];
        problem.Aineq = (double*) A_ineq_buffer_info.ptr;
        py::buffer_info b_ineq_buffer_info  = b_ineq.cast<py::buffer>().request();
        if (b_ineq_buffer_info.format != "d")
        {
          throw std::invalid_argument("b_ineq must be a double array");
        }
        problem.bineq = (double*) b_ineq_buffer_info.ptr;
      }
      

      // Initialize the result, call the function, convert the return type, and return it.
      prima_result_t result;
      const prima_rc_t rc = prima_minimize(algorithm, &problem, &options, &result);
      PRIMAResult result_copy(result, py_x0.size(), problem.m_nlcon);
      prima_free_result(&result);
      return result_copy;
    }, "fun"_a, "x0"_a, "args"_a=py::tuple(), "method"_a=py::none(),
           "lb"_a=py::none(), "ub"_a=py::none(), "A_eq"_a=py::none(), "b_eq"_a=py::none(),
           "A_ineq"_a=py::none(), "b_ineq"_a=py::none(),
           "constraint_function"_a=py::none(),
           "callback"_a=nullptr, "options"_a=py::none()
  );
}
