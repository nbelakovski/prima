// Dedicated to the late Professor M. J. D. Powell FRS (1936--2015).


#include "prima/prima.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 * A NOTE ON DEFAULT VALUES IN OPTIONS AND PROBLEM STRUCTURES
 *
 * Certain values of the variables in the options and problems structures
 * are interpreted by the Fortran code as "not present". This is not by default,
 * it is done by us intentionally so that we may signal to the Fortran code that
 * these values were not provided. This is so that the Fortran code may then properly
 * set the default values for those variables.
 *
 * In order to accomplish this we take advantage of a certain part of the Fortran
 * standard that basically says that if an allocatable value which has not been
 * allocated is passed to a procedure, `present` will return false.
 *
 * Our convention is as follows
 * double  - NaN  is interpreted as not present
 * int     - 0    is interpreted as not present (as of 20240124 all ints are expected nonnegative)
 * pointer - NULL is interpreted as not present
 *
 * If variables are added to options/problems that are optional, the algorithm_c.f90 files must
 * be updated to treat the default values appropriately. For examples see rhobeg/rhoend(double),
 * maxfun/npt(int), and xl/xu (array/pointer).
 */


// Function to initialize the problem
int prima_init_problem(prima_problem_t *problem, int n)
{
    if (!problem)
        return PRIMA_NULL_PROBLEM;

    memset(problem, 0, sizeof(prima_problem_t));
    problem->n = n;
    problem->f0 = NAN;
    return 0;
}


// Function to initialize the options
int prima_init_options(prima_options_t *options)
{
    if (!options)
        return PRIMA_NULL_OPTIONS;

    memset(options, 0, sizeof(prima_options_t));
    options->rhobeg = NAN;  // Will be interpreted by Fortran as not present
    options->rhoend = NAN;  // Will be interpreted by Fortran as not present
    options->iprint = PRIMA_MSG_NONE;
    options->ftarget = -INFINITY;
    return 0;
}


// Function to check whether the problem matches the algorithm
prima_rc_t prima_check_problem(prima_problem_t *problem, prima_options_t *options, const int use_constr)
{
    if (!problem)
        return PRIMA_NULL_PROBLEM;

    if (!options)
        return PRIMA_NULL_OPTIONS;

    if (!problem->x0)
        return PRIMA_NULL_X0;

    if ((use_constr && !problem->calcfc) || (!use_constr && !problem->calfun))
        return PRIMA_NULL_FUNCTION;

    return 0;
}


// Function to initialize the result
prima_rc_t prima_init_result(prima_result_t *result, prima_problem_t *problem)
{
    if (!result)
        return PRIMA_NULL_RESULT;

    memset(result, 0, sizeof(prima_result_t));

    if (!problem)
        return PRIMA_NULL_PROBLEM;

    // x: returned point
    result->x = (double*)malloc(problem->n * sizeof(double));
    if (!result->x)
        return PRIMA_MEMORY_ALLOCATION_FAILS;
    for (int i = 0; i < problem->n; i++)
        result->x[i] = problem->x0[i];

    // f: objective function value at the returned point
    result->f = NAN;

    // cstrv: constraint violation at the returned point (COBYLA and LINCOA only)
    result->cstrv = NAN;

    // nlconstr: nonlinear constraint values at the returned point, of size m_nlcon (COBYLA only)
    if (problem->m_nlcon <= 0)
        result->nlconstr = NULL;
    else {
        result->nlconstr = (double*)malloc(problem->m_nlcon * sizeof(double));
        if (!result->nlconstr) {
            free(result->x);
            return PRIMA_MEMORY_ALLOCATION_FAILS;
        }
        for (int i = 0; i < problem->m_nlcon; i++)
            result->nlconstr[i] = problem->nlconstr0 ? problem->nlconstr0[i] : NAN;
    }

    // nf: number of function evaluations
    result->nf = 0;

    // status: return code
    result->status = PRIMA_RESULT_INITIALIZED;

    // message: exit message
    result->message = NULL;

    return 0;
}


// Function to free the result
int prima_free_result(prima_result_t *result)
{
    if (!result)
        return PRIMA_NULL_RESULT;

    if (result->nlconstr) {
        free(result->nlconstr);
        result->nlconstr = NULL;
    }
    if (result->x) {
        free(result->x);
        result->x = NULL;
    }
    return 0;
}


// Function to get the string corresponding to the return code
const char *prima_get_rc_string(const prima_rc_t rc)
{
    switch (rc) {
        case PRIMA_SMALL_TR_RADIUS:
            return "Trust region radius reaches its lower bound";
        case PRIMA_FTARGET_ACHIEVED:
            return "The target function value is reached";
        case PRIMA_TRSUBP_FAILED:
            return "A trust region step failed to reduce the model";
        case PRIMA_MAXFUN_REACHED:
            return "Maximum number of function evaluations reached";
        case PRIMA_MAXTR_REACHED:
            return "Maximum number of trust region iterations reached";
        case PRIMA_NAN_INF_X:
            return "The input X contains NaN of Inf";
        case PRIMA_NAN_INF_F:
            return "The objective or constraint functions return NaN or +Inf";
        case PRIMA_NAN_INF_MODEL:
            return "NaN or Inf occurs in the model";
        case PRIMA_NO_SPACE_BETWEEN_BOUNDS:
            return "No space between bounds";
        case PRIMA_DAMAGING_ROUNDING:
            return "Rounding errors are becoming damaging";
        case PRIMA_ZERO_LINEAR_CONSTRAINT:
            return "One of the linear constraints has a zero gradient";
        case PRIMA_CALLBACK_TERMINATE:
            return "Callback function requested termination of optimization";
        case PRIMA_INVALID_INPUT:
            return "Invalid input";
        case PRIMA_ASSERTION_FAILS:
            return "Assertion fails";
        case PRIMA_VALIDATION_FAILS:
            return "Validation fails";
        case PRIMA_MEMORY_ALLOCATION_FAILS:
            return "Memory allocation fails";
        case PRIMA_NULL_OPTIONS:
            return "NULL options";
        case PRIMA_NULL_PROBLEM:
            return "NULL problem";
        case PRIMA_NULL_X0:
            return "NULL x0";
        case PRIMA_NULL_RESULT:
            return "NULL result";
        case PRIMA_NULL_FUNCTION:
            return "NULL function";
        default:
            return "Invalid return code";
    }
}


// Functions implemented in Fortran (*_c.f90)
int cobyla_c(const int m_nlcon, const prima_objcon_t calcfc, const void *data, const int n, double x[], double *f, double *cstrv, double nlconstr[],
            const int m_ineq, const double Aineq[], const double bineq[],
            const int m_eq, const double Aeq[], const double beq[],
            const double xl[], const double xu[],
            const double f0, const double nlconstr0[],
            int *nf, const double rhobeg, const double rhoend, const double ftarget, const int maxfun, const int iprint, const prima_callback_t callback, int *info);

int bobyqa_c(prima_obj_t calfun, const void *data, const int n, double x[], double *f, const double xl[], const double xu[],
            int *nf, const double rhobeg, const double rhoend, const double ftarget, const int maxfun, const int npt, const int iprint, const prima_callback_t callback, int *info);

int newuoa_c(prima_obj_t calfun, const void *data, const int n, double x[], double *f,
            int *nf, const double rhobeg, const double rhoend, const double ftarget, const int maxfun, const int npt, const int iprint, const prima_callback_t callback, int *info);

int uobyqa_c(prima_obj_t calfun, const void *data, const int n, double x[], double *f,
            int *nf, const double rhobeg, const double rhoend, const double ftarget, const int maxfun, const int iprint, const prima_callback_t callback, int *info);

int lincoa_c(prima_obj_t calfun, const void *data, const int n, double x[], double *f,
            double *cstrv,
            const int m_ineq, const double Aineq[], const double bineq[],
            const int m_eq, const double Aeq[], const double beq[],
            const double xl[], const double xu[],
            int *nf, const double rhobeg, const double rhoend, const double ftarget, const int maxfun, const int npt, const int iprint, const prima_callback_t callback, int *info);


// The function that does the minimization using a PRIMA solver
prima_rc_t prima_minimize(const prima_algorithm_t algorithm, prima_problem_t *problem, prima_options_t *options, prima_result_t *result)
{
    int use_constr = (algorithm == PRIMA_COBYLA);

    prima_rc_t info = prima_check_problem(problem, options, use_constr);

    if (info == 0)
        info = prima_init_result(result, problem);

    if (info == 0) {
        switch (algorithm) {
            case PRIMA_BOBYQA:
                bobyqa_c(problem->calfun, options->data, problem->n, result->x, &(result->f), problem->xl, problem->xu, &(result->nf), options->rhobeg, options->rhoend, options->ftarget, options->maxfun, options->npt, options->iprint, options->callback, &info);
                result->cstrv = 0.0;
                break;

            case PRIMA_COBYLA:
                cobyla_c(problem->m_nlcon, problem->calcfc, options->data, problem->n, result->x, &(result->f), &(result->cstrv), result->nlconstr,
                            problem->m_ineq, problem->Aineq, problem->bineq, problem->m_eq, problem->Aeq, problem->beq,
                            problem->xl, problem->xu, problem->f0, problem->nlconstr0, &(result->nf), options->rhobeg, options->rhoend, options->ftarget, options->maxfun, options->iprint, options->callback, &info);
                break;

            case PRIMA_LINCOA:
                lincoa_c(problem->calfun, options->data, problem->n, result->x, &(result->f), &(result->cstrv),
                            problem->m_ineq, problem->Aineq, problem->bineq, problem->m_eq, problem->Aeq, problem->beq,
                            problem->xl, problem->xu, &(result->nf), options->rhobeg, options->rhoend, options->ftarget, options->maxfun, options->npt, options->iprint, options->callback, &info);
                break;

            case PRIMA_NEWUOA:
                newuoa_c(problem->calfun, options->data, problem->n, result->x, &(result->f), &(result->nf), options->rhobeg, options->rhoend, options->ftarget, options->maxfun, options->npt, options->iprint, options->callback, &info);
                result->cstrv = 0.0;
                break;

            case PRIMA_UOBYQA:
                uobyqa_c(problem->calfun, options->data, problem->n, result->x, &(result->f), &(result->nf), options->rhobeg, options->rhoend, options->ftarget, options->maxfun, options->iprint, options->callback, &info);
                result->cstrv = 0.0;
                break;

            default:
                return PRIMA_INVALID_INPUT;
        }

        result->status = info;
        result->message = prima_get_rc_string(info);
    }

    return info;
}

bool prima_is_success(const prima_result_t result)
{
    return (result.status == PRIMA_SMALL_TR_RADIUS ||
            result.status == PRIMA_FTARGET_ACHIEVED) && (result.cstrv <= sqrt(DBL_EPSILON));
}
