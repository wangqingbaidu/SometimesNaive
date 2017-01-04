/*
 * LSTM.h
 *
 *  Created on: Dec 27, 2016
 *      Author: wangqingbaidu
 */

#ifndef LSTM_H_
#define LSTM_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "mxnet-cpp/MxNetCpp.h"
#include "memory.h"

using namespace std;
using namespace mxnet::cpp;

class LSTM {
public:
	LSTM():
		ctx(Context(DeviceType::kCPU, 0)),
		bf(Symbol::Variable("bf")),
		bi(Symbol::Variable("bi")),
		bo(Symbol::Variable("bo")),
		bc(Symbol::Variable("bc")),
		uf(Symbol::Variable("uf")),
		ui(Symbol::Variable("ui")),
		uo(Symbol::Variable("uo")),
		uc(Symbol::Variable("uc")),

		fc_w(Symbol::Variable("fc_w")),
		fc_b(Symbol::Variable("fc_b")){}
	LSTM(int d, int h, int v, const char* fname = NULL):
		D(d), H(h), V(v),
		ctx(Context(DeviceType::kCPU, 0)),
		wf(Symbol::Variable("wf")),
		wi(Symbol::Variable("wi")),
		wo(Symbol::Variable("wo")),
		wc(Symbol::Variable("wc")),

		bf(Symbol::Variable("bf")),
		bi(Symbol::Variable("bi")),
		bo(Symbol::Variable("bo")),
		bc(Symbol::Variable("bc")),

		uf(Symbol::Variable("uf")),
		ui(Symbol::Variable("ui")),
		uo(Symbol::Variable("uo")),
		uc(Symbol::Variable("uc")),

		fc_w(Symbol::Variable("fc_w")),
		fc_b(Symbol::Variable("fc_b"))
	{
		args_map["ht"] = NDArray(Shape(1,H), ctx, false);
		args_map["ct"] = NDArray(Shape(1,H), ctx, false);
		args_map["ht"] = 0.0f;
		args_map["ct"] = 0.0f;
		if (fname)
		{
			string s(fname);
			if (! load_data(s))
				exit(0);
		}
	}
	virtual ~LSTM(){}
	void reset_state()
	{
		args_map["ht"] = 0.0f;
		args_map["ct"] = 0.0f;
	}

	NDArray get_output(mx_float* data)
	{
		NDArray input_data(data, Shape(1, D), ctx);
		Symbol ht = Symbol::Variable("ht");
		Symbol ct = Symbol::Variable("ct");
		args_map["data"] = input_data;
		//Generate ct executor
		Symbol input = Symbol::Variable("data");
		Symbol ft =  dot(input, wf) + dot(ht, uf) + bf;
		Symbol a_ft = Activation(ft, ActivationActType::sigmoid);

		Symbol it =  dot(input, wi) + dot(ht, ui) + bi;
		Symbol a_it = Activation(it, ActivationActType::sigmoid);

		Symbol tmp = dot(input, wc) + dot(ht, uc) + bc;
		Symbol a_tmp = Activation(tmp, ActivationActType::tanh);
		Symbol new_ct =  a_ft * ct + a_it * a_tmp;

		Executor* ct_exe = new_ct.SimpleBind(ctx, args_map);
		ct_exe->Forward(false);
		NDArray::WaitAll();
		NDArray ct_data = ct_exe->outputs[0];
		args_map["ct"] = ct_data;

		//Generate ht executor
		Symbol ot = dot(input, wo) + dot(ht, uo) + bo;
		Symbol a_ot = Activation(ot, ActivationActType::sigmoid);
		Executor* ot_exe = a_ot.SimpleBind(ctx, args_map);
		ot_exe->Forward(false);
		NDArray::WaitAll();

		Symbol a_ct = Activation(ct, ActivationActType::tanh);
		Executor* act_exe = a_ct.SimpleBind(ctx, args_map);
		act_exe->Forward(false);
		NDArray::WaitAll();
		args_map["ht"] =  (ot_exe->outputs[0] * act_exe->outputs[0]);
		args_map["rht"] = args_map["ht"].Reshape(Shape(H, 1));
		
		//Generate output executor
		Symbol rht = Symbol::Variable("rht");
		Symbol fc = dot(fc_w, rht) +fc_b;
		Executor* output_exe = fc.SimpleBind(ctx, args_map);
		output_exe->Forward(false);
		NDArray::WaitAll();
		auto rct = output_exe->outputs[0];
		delete ct_exe;
		delete ot_exe;
		delete output_exe;
		return rct;
	}

	int load_data(string fname)
	{
		char* buffer = 0;
		mx_float* buffer_ptr = 0;
		int total_bytes = 0;
		ifstream in(fname, ios::binary);
		if (! in.is_open())
		{
			cout << fname << " file not exists!" << endl;
			return 0;
		}
		/*---------------PHASE ONE: Read file data to buffer----------------*/
		total_bytes = (4 * (D * H + H * H + H) + H * V + V) * sizeof(mx_float);
		buffer = (char*)calloc(total_bytes, sizeof(char));
		in.read(buffer, total_bytes);
		in.close();
		buffer_ptr = (mx_float*)buffer;

		/*---------------PHASE TWO: Initialize weights NDArray----------------*/
		//read wi data
		NDArray wi_data(buffer_ptr, Shape(D, H), ctx);
		buffer_ptr += D*H;
		//read ui data
		NDArray ui_data(buffer_ptr, Shape(H, H), ctx);
		buffer_ptr += H*H;
		//read bi data
		NDArray bi_data(buffer_ptr, Shape(1, H), ctx);
		buffer_ptr += H;

		//read wf data
		NDArray wf_data(buffer_ptr, Shape(D, H), ctx);
		buffer_ptr += D*H;
		//read uf data
		NDArray uf_data(buffer_ptr, Shape(H, H), ctx);
		buffer_ptr += H*H;
		//read bf data
		NDArray bf_data(buffer_ptr, Shape(1, H), ctx);
		buffer_ptr += H;

		//read wo data
		NDArray wo_data(buffer_ptr, Shape(D, H), ctx);
		buffer_ptr += D*H;
		//read uo data
		NDArray uo_data(buffer_ptr, Shape(H, H), ctx);
		buffer_ptr += H*H;
		//read bo data
		NDArray bo_data(buffer_ptr, Shape(1, H), ctx);
		buffer_ptr += H;

		//read wc data
		NDArray wc_data(buffer_ptr, Shape(D, H), ctx);
		buffer_ptr += D*H;
		//read uc data
		NDArray uc_data(buffer_ptr, Shape(H, H), ctx);
		buffer_ptr += H*H;
		//read bc data
		NDArray bc_data(buffer_ptr, Shape(1, H), ctx);
		buffer_ptr += H;

		//read fc weights data
		NDArray fc_w_data(buffer_ptr, Shape(V, H), ctx);
		buffer_ptr += H*V;
		//read fc bias data
		NDArray fc_b_data(buffer_ptr, Shape(V, 1), ctx);

		/*---------------PHASE THREE: Set data context----------------*/
		args_map["wi"] = wi_data;
		args_map["ui"] = ui_data;
		args_map["bi"] = bi_data;
		args_map["wf"] = wf_data;
		args_map["uf"] = uf_data;
		args_map["bf"] = bf_data;
		args_map["wo"] = wo_data;
		args_map["uo"] = uo_data;
		args_map["bo"] = bo_data;
		args_map["wc"] = wc_data;
		args_map["uc"] = uc_data;
		args_map["bc"] = bc_data;
		args_map["fc_w"] = fc_w_data;
		args_map["fc_b"] = fc_b_data;

		/*---------------PHASE FOUR: Collect garbage and initialize ht, ct----------------*/
		free(buffer);
		args_map["ht"] = 0;
		args_map["ct"] = 0;
		return 1;
	}

private:
	Symbol wi, ui, bi;
	Symbol wf, uf, bf;
	Symbol wo, uo, bo;
	Symbol wc, uc, bc;
	Symbol fc_w, fc_b;
	Context ctx;
	map<string, NDArray> args_map;
private:
	int D;
	int H;
	int V;
};

#endif /* LSTM_H_ */
