#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Core>

namespace femto {

typedef float Real;
typedef Eigen::Matrix<Real,-1,1> Vector;
typedef Eigen::Matrix<Real,-1,-1> Matrix;

typedef std::pair<Vector,Vector> Sample;

struct MLP {
    struct Layer {
        Matrix W;   // Weights with the previous layer
        Vector b;   // biases
        std::pointer_to_unary_function<Real,Real> f,df; // activation functions and derivative
        mutable Vector y;

        Layer(int prevLayerSize, int thisLayerSize, std::pointer_to_unary_function<Real,Real> _f, std::pointer_to_unary_function<Real,Real> _df) :
            W(Matrix::Random(thisLayerSize,prevLayerSize)),
            b(Vector::Random(thisLayerSize)),
            f(_f), df(_df),
            y(thisLayerSize) {}

        int nInputs() const { return W.cols(); }
        int nOutputs() const { return W.rows(); }
    };
    int nFeats;
    std::vector<Layer> layers; // layers

    MLP(int n) : nFeats(n) {}

    template<typename Activation>
    MLP & addLayer(int nNeurons) {
        int n0 = layers.empty() ? nFeats : nOutputs();
        layers.emplace_back(n0, nNeurons, std::ptr_fun(Activation::f), std::ptr_fun(Activation::df) );
        return *this;
    }

    int nFeatures() const { return nFeats; }
    int nOutputs(int iLayer=-1) const { return iLayer<0 ? layers.back().nOutputs() : layers[iLayer].nOutputs(); }

    const Vector & operator()(const Vector & x) const { return eval(x); }
    const Vector & eval(const Vector & x) const {
        layers[0].y = (layers[0].W*x + layers[0].b).unaryExpr(layers[0].f);
        for(int iLayer=1; iLayer<layers.size(); ++iLayer)
            layers[iLayer].y = (layers[iLayer].W*layers[iLayer-1].y + layers[iLayer].b).unaryExpr(layers[iLayer].f);
        return layers.back().y;
    }
    Real train(const Matrix & xx, const Matrix & yy, Real lr=0.1, size_t nIts=std::numeric_limits<size_t>::max(), int batch=-1) {
        assert(xx.cols()==yy.cols());
        assert(xx.rows()==nFeats);
        assert(yy.rows()==nOutputs());

        const int N = xx.cols();

        const int nLayers = layers.size();

        Real cost;

        if(batch<0)
            batch = N;

        // local data preallocated
        Eigen::Array<Matrix,-1,1> dCdW(nLayers);
        Eigen::Array<Vector,-1,1> dCdb(nLayers);
        Eigen::Array<Vector,-1,1> a(nLayers);
        Eigen::Array<Vector,-1,1> z(nLayers);
        Eigen::Array<Vector,-1,1> delta(nLayers);
        for(int iLayer=0; iLayer<nLayers; ++iLayer) {
            dCdW[iLayer] .resize(layers[iLayer].W.rows(), layers[iLayer].W.cols());
            dCdb[iLayer] .resize(layers[iLayer].b.rows(), 1);
            a[iLayer]    .resize(layers[iLayer].b.rows(), 1);
            z[iLayer]    .resize(layers[iLayer].b.rows(), 1);
            delta[iLayer].resize(layers[iLayer].b.rows(), 1);
        }

        for(int it=1; it<=nIts; ++it) {

            // reset Weights and biases derivatives
            for(int iLayer=0; iLayer<nLayers; ++iLayer) {
                dCdW[iLayer].setZero();
                dCdb[iLayer].setZero();
            }

            cost = 0;
            for(int iBatch=0; iBatch<batch; ++iBatch) {
                int iSample = batch<N ? rand()%N : iBatch;

                // feed forward
                a[0] = (z[0] = layers[0].W*xx.col(iSample) + layers[0].b).unaryExpr( layers[0].f );
                for(int iLayer=1; iLayer<nLayers; ++iLayer)
                    a[iLayer] = (z[iLayer] = layers[iLayer].W*a[iLayer-1] + layers[iLayer].b).unaryExpr( layers[iLayer].f );

                // backpropagation
                delta[nLayers-1] = a[nLayers-1] - yy.col(iSample);
                cost += delta[nLayers-1].norm();
                for(int iLayer=nLayers-2; iLayer>=0; --iLayer) {
                    dCdW[iLayer+1] += delta[iLayer+1] * a[iLayer].transpose();
                    dCdb[iLayer+1] += delta[iLayer+1];
                    delta[iLayer] = (layers[iLayer+1].W.transpose() * delta[iLayer+1]).cwiseProduct(z[iLayer].unaryExpr( layers[iLayer].df ));
                }
                dCdW[0] += delta[0] * xx.col(iSample).transpose();
                dCdb[0] += delta[0];
            }

            cost /= batch;
            if(it%10000==0)
                std::cout << it << "\tcost\t" << cost << std::endl;

            // gradient descent
            for(int iLayer=0; iLayer<nLayers; ++iLayer) {
                layers[iLayer].W -= dCdW[iLayer]*(lr/batch)/(1+std::log(it));
                layers[iLayer].b -= dCdb[iLayer]*(lr/batch)/(1+std::log(it));
            }

        }

        return cost;
    }

};

struct Sigmoid {
    static Real f (Real x) { return 1./(1. + std::exp(-x)); }
    static Real df(Real x) { x = f(x); return x*(1.-x); }
};
struct Identity {
    static Real f (Real x) { return x; }
    static Real df(Real x) { return 1; }
};
struct Cubic {
    static Real f (Real x) { return x<0 ? 0 : (x>1 ? 1 : x*x*(3 - 2*x)); }
    static Real df(Real x) { return (x<0 || x>1) ? 0 : 6*x*(1-x); }
};
struct Quintic {
    static Real f (Real x) { return x<0 ? 0 : (x>1 ? 1 : x*x*x*(10+x*(6*x-15))); }
    static Real df(Real x) { return (x<0 || x>1) ? 0 : 30*x*x*(x-1)*(x-1); }
};
struct Ramp {
    static Real f (Real x) { return x<0 ? 0 : (x>1 ? 1 : x); }
    static Real df(Real x) { return (x<0||x>1) ? 0 : 1; }
};
struct Relu {
    static Real f (Real x) { return (x<0) ? 0 : x; }
    static Real df(Real x) { return (x<0) ? 0 : 1; }
};

}   // namespace mlp
