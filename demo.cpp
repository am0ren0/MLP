#include "MLP.h"
using namespace femto;

using namespace std;

#include <Eigen/Dense>


int main(/*int argc, char *argv[]*/) {
    srand( time(0) );
    MLP nn(1);

    nn.
            addLayer<Sigmoid>(50).
            addLayer<Sigmoid>(50).
            addLayer<Identity>(1);

    for(auto & layer: nn.layers) cout << layer.nInputs() << "-->" << layer.nOutputs() << ", ";
    cout << endl;


    int nSamples = 15000;

    Matrix xx = Matrix::Random(nn.nFeats, nSamples)*M_PI;
    Matrix yy = xx.array().sin().matrix();

    nn.train(xx,yy,.1, 100000,50);

    Vector x(1);
    int n=100;
    for(int i=-n; i<=n; ++i) {
        x[0] = i*M_PI/n;
        Real y = std::sin(x[0]);
        cout << x << "\t" << y << "\t" << nn.eval(x) << endl;
    }
    for(const auto & lyr: nn.layers) {
        cout << "W: " << endl << lyr.W << endl;
        cout << "b: " << endl << lyr.b << endl;
    }

    return 0;
}
