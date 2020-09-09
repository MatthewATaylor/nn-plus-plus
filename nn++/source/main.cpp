#include <iostream>

#include "Activation.h"
#include "Loss.h"
#include "Dense.h"
#include "Network.h"
#include "math/Vec.h"

int main() {
	Dense<2, 4, Activation::ReLU> dense1;
	Dense<4, 4, Activation::ReLU> dense2;
	Dense<4, 1, Activation::Linear> dense3;
	Network network(
		&dense1,
		&dense2,
		&dense3
	);

	Vec<float, 2> input(1.0f, 2.0f);
	Vec<float, 4> a1 = dense1.evaluate(input);
	Vec<float, 4> a2 = dense2.evaluate(a1);
	Vec<float, 1> output = dense3.evaluate(a2);
	std::cout << output << "\n";

	Vec<float, 1> target(3.0f);
	std::cout << network.getLoss<2, 1, Loss::MeanSquaredError>(input, target) << "\n";

	std::cin.get();
	return 0;
}
