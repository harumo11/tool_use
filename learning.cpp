#include <iostream>
#include <vector>
#include <random>
#include <fanda/Csv.hpp>
#include <Gpop/Series.hpp>
#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/io.h>

int main(int argc, char* argv[])
{
	
	//dynetの初期化
	dynet::initialize(argc, argv);

	//設定
	const int INPUT_SIZE = 19;
	const int HIDDEN_SIZE = 19;
	const int OUTPUT_SIZE = 7;
	const int MINIBATCH_SIZE = 32;
	const int ITERATIONS = 100;

	//計算グラフを構築
	dynet::ComputationGraph cg;
	dynet::ParameterCollection model;
	dynet::SimpleSGDTrainer trainer(model);

	//パラメータを設定
	dynet::Parameter p_W1 = model.add_parameters({INPUT_SIZE, HIDDEN_SIZE});
	dynet::Parameter p_b1 = model.add_parameters({HIDDEN_SIZE});
	dynet::Parameter p_W2 = model.add_parameters({HIDDEN_SIZE, HIDDEN_SIZE});
	dynet::Parameter p_b2 = model.add_parameters({HIDDEN_SIZE});
	dynet::Parameter p_W3 = model.add_parameters({OUTPUT_SIZE, HIDDEN_SIZE});
	dynet::Parameter p_b3 = model.add_parameters({OUTPUT_SIZE});

	//関係性を定義
	dynet::Expression W1 = dynet::parameter(cg, p_W1);
	dynet::Expression b1 = dynet::parameter(cg, p_b1);
	dynet::Expression W2 = dynet::parameter(cg, p_W2);
	dynet::Expression b2 = dynet::parameter(cg, p_b2);
	dynet::Expression W3 = dynet::parameter(cg, p_W3);
	dynet::Expression b3 = dynet::parameter(cg, p_b3);

	//入出力を定義
	std::vector<dynet::real> x_value(INPUT_SIZE*MINIBATCH_SIZE);
	std::vector<dynet::real> y_value(OUTPUT_SIZE*MINIBATCH_SIZE);
	dynet::Dim x_dim({INPUT_SIZE}, MINIBATCH_SIZE);
	dynet::Dim y_dim({OUTPUT_SIZE}, MINIBATCH_SIZE);
	dynet::Expression x = dynet::input(cg, x_dim, &x_value);
	dynet::Expression y = dynet::input(cg, y_dim, &y_value);


	//ノードの関係性を定義
	std::cout << "D0" << std::endl;
	dynet::Expression h1 = dynet::rectify(W1*x  + b1);
	std::cout << "D1" << std::endl;
	dynet::Expression h2 = dynet::rectify(W2*h1 + b2);
	std::cout << "D2" << std::endl;
	std::cout << "h2 : " << h2.dim() << std::endl;
	std::cout << "W3 : " << W3.dim() << std::endl;
	std::cout << "b3 : " << b3.dim() << std::endl;
	dynet::Expression y_pred = W3*h2 + b3;
	std::cout << "D3" << std::endl;

	dynet::Expression loss_expr = dynet::squared_distance(y_pred, y);
	dynet::Expression last = dynet::sum_batches(loss_expr);

	cg.print_graphviz();

	//ミニバッチデータ用意
	CSV::CsvFile csv("/home/harumo/catkin_ws/src/sia20/sia20_control/data/log.csv");
	if (!csv.is_open()) {
		std::cout << "[ERROR] Can't open csv data file" << std::endl;
	}
	else {
		std::cout << "[INFO] Opened csv data file" << std::endl;
		std::cout << "[INFO] csv raw size : " << csv.raw_size() << std::endl;
		std::cout << "[INFO] csv collumn size : " << csv.collumn_size() << std::endl;
	}

	//グラフ準備
	std::vector<double> loss_vec;
	
	//学習フェーズ
	for (int i = 0; i < ITERATIONS; i++) {
		std::cout << "[info] iter : " << i << std::endl;

		//入力ベクタークリア
		x_value.clear();
		y_value.clear();
	
		//ミニバッチ作成
		//ランダムインデックス作成
		std::random_device rand;
		std::uniform_int_distribution<> filter(0, csv.raw_size()-1);
	
		for (int i = 0; i < MINIBATCH_SIZE; i++) {
			const int random_index = filter(rand);
	
			//ミニバッチ化された入力
			for (int j = 0; j < INPUT_SIZE; j++) {
				x_value.push_back(csv(random_index, j).get_as_double());
			}
			//ミニバッチ化された出力
			for (int k = INPUT_SIZE; k < INPUT_SIZE+OUTPUT_SIZE; k++) {
				std::cout << k << " : " << csv(random_index, k).get_as_double() << std::endl;
				y_value.push_back(csv(random_index, k).get_as_double());
			}
		}
	
		std::cout << "[info] minibatch_x size : " << MINIBATCH_SIZE*INPUT_SIZE << " : " << x_value.size() << std::endl;
		std::cout << "[info] minibatch_y size : " << MINIBATCH_SIZE*OUTPUT_SIZE << " : " << y_value.size() << std::endl;
	
		//foward prop
		float loss = dynet::as_scalar(cg.forward(last));
		std::cout << "[info] loss : " << loss << std::endl;
		loss_vec.push_back(loss);
	
		//back prop
		cg.backward(last);
	
		//trainer update
		trainer.update();
	}

	//グラフ描画
	Gpop::Series plot;
	plot.plot(loss_vec);
	plot.show();
	std::cin.get();
	plot.save_as_png("loss");

	//loss保存
	std::ofstream log_loss("loss.txt");
	for (auto e : loss_vec){
		log_loss << e << std::endl;
	}

	//model保存
	dynet::TextFileSaver model_saver("tool1.model");
	model_saver.save(model);

	return 0;
}
