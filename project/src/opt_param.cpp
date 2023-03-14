#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;


struct CostResidual{
    CostResidual(double theta, double tau): theta_(theta), tau_(tau) {}

    template <typename T>
    bool operator()(const T* const d, const T* const a, const T* const b, const T* const k, T* residual) const
    {
        residual[0] = tau_ - k[0]*d[0]*
        (
            sqrt(d[0]*d[0] + 2.0*d[0]*b[0]+a[0]*a[0]+b[0]*b[0])
            -sqrt(d[0]*d[0] -2.0*d[0]*a[0]*sin(theta_) + 2.0*d[0]*b[0]*cos(theta_)+a[0]*a[0]+b[0]*b[0])
        )
        /sqrt(d[0]*d[0] - 2.0*d[0]*a[0]*sin(theta_) + 2.0*d[0]*b[0]*cos(theta_)+a[0]*a[0]+b[0]*b[0])
        *(a[0]*cos(theta_)+b[0]*sin(theta_));
        return true;
    }

    private:
        const double theta_;
        const double tau_;
};


int main(int argc, char**argv)
{
    double data[90*2];

    google::InitGoogleLogging(argv[0]);

    double a = 0.001;
    double b = 0.001;
    double d = 0.2;
    double k = 10e3;


    int N = 90;

    for(int i = 0; i < N;i++)
    {
        data[2*i] = (double) i*M_PI/180.0;
        data[2*i+1] = (double) 100.0*sin(i*M_PI/180.0);
    }

    std::cout<<"Start\n";

    Problem problem;


    for(int i = 0; i < N; i++)
    {
        problem.AddResidualBlock(new AutoDiffCostFunction<CostResidual,1,1,1,1,1>
        (new CostResidual(data[2*i], data[2*i+1])),nullptr,&d,&a,&b,&k);
    }

    problem.SetParameterLowerBound(&a,0,0.0);
    problem.SetParameterUpperBound(&a,0,0.1);
    
    problem.SetParameterLowerBound(&b,0,0.0);
    problem.SetParameterUpperBound(&b,0,0.1);

    problem.SetParameterLowerBound(&d,0,0.20);
    problem.SetParameterUpperBound(&d,0,0.40);

    problem.SetParameterLowerBound(&k,0,1e3);
    problem.SetParameterUpperBound(&k,0,50e3);

    Solver::Options options;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-5;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout<< summary.BriefReport() <<"\n";
    std::cout<<"a: "<<a<<"\n";
    std::cout<<"b: "<<b<<"\n";
    std::cout<<"d: "<<d<<"\n";
    std::cout<<"k: "<<k<<"\n";

    return EXIT_SUCCESS;
}
