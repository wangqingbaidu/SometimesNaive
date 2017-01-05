/*
 * LSTM.h
 *
 *  Created on: Dec 27, 2016
 *      Author: wangqingbaidu
 */

#ifndef LSTM_NO_MXNET_H_
#define LSTM_NO_MXNET_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <memory.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

class LSTM {
public:
	LSTM(){
		this->D = 0;
		this->H = 0;
		this->V = 0;
	}
	LSTM(int d, int h, int v, const char* fname = NULL):D(d), H(h), V(v)
	{
		if (fname)
		{
			string s(fname);
			if (! load_data(s))
				exit(0);
			reset_state();
		}
	}
	virtual ~LSTM()
	{
		free(data);
		free(args_map["ht"]);
		free(args_map["ct"]);

		free(args_map["temp1_H"]);
		free(args_map["temp2_H"]);
		free(args_map["a_ft"]);
		free(args_map["a_it"]);
		free(args_map["a_tmp"]);
		free(args_map["a_ot"]);
		free(args_map["a_ct"]);
	}
	void reset_state()
	{
		memset(args_map["ht"], 0, H*sizeof(float));
		memset(args_map["ct"], 0, H*sizeof(float));
	}

	vector<float> get_output(float* data)
	{
//		NDArray input_data(data, Shape(1, D), ctx);
//		Symbol ht = Symbol::Variable("ht");
//		Symbol ct = Symbol::Variable("ct");
//		args_map["data"] = input_data;
//		//Generate ct executor
//		Symbol input = Symbol::Variable("data");

		matrix_mul(data, args_map["wf"], args_map["temp1_H"], 1, D, H);
		matrix_mul(args_map["ht"], args_map["uf"], args_map["temp2_H"], 1, H, H);
		matrix_add(args_map["temp1_H"], args_map["temp2_H"], args_map["a_ft"], H);
		matrix_add(args_map["a_ft"], args_map["bf"], args_map["a_ft"], H);
		matrix_sigmoid(args_map["a_ft"], args_map["a_ft"], H);

//		Symbol ft =  dot(input, wf) + dot(ht, uf) + bf;
//		Symbol a_ft = Activation(ft, ActivationActType::sigmoid);

		matrix_mul(data, args_map["wi"], args_map["temp1_H"], 1, D, H);
		matrix_mul(args_map["ht"], args_map["ui"], args_map["temp2_H"], 1, H, H);
		matrix_add(args_map["temp1_H"], args_map["temp2_H"], args_map["a_it"], H);
		matrix_add(args_map["a_it"], args_map["bi"], args_map["a_it"], H);
		matrix_sigmoid(args_map["a_it"], args_map["a_it"], H);

//		Symbol it =  dot(input, wi) + dot(ht, ui) + bi;
//		Symbol a_it = Activation(it, ActivationActType::sigmoid);

		matrix_mul(data, args_map["wc"], args_map["temp1_H"], 1, D, H);
		matrix_mul(args_map["ht"], args_map["uc"], args_map["temp2_H"], 1, H, H);
		matrix_add(args_map["temp1_H"], args_map["temp2_H"], args_map["a_tmp"], H);
		matrix_add(args_map["a_tmp"], args_map["bc"], args_map["a_tmp"], H);
		matrix_tanh(args_map["a_tmp"], args_map["a_tmp"], H);
		matrix_mul_elementwise(args_map["a_ft"], args_map["ct"], args_map["temp1_H"], H);
		matrix_mul_elementwise(args_map["a_it"], args_map["a_tmp"], args_map["temp2_H"], H);
		matrix_add(args_map["temp1_H"], args_map["temp2_H"], args_map["ct"], H);

//		Symbol tmp = dot(input, wc) + dot(ht, uc) + bc;
//		Symbol a_tmp = Activation(tmp, ActivationActType::tanh);
//		Symbol new_ct =  a_ft * ct + a_it * a_tmp;

//		Executor* ct_exe = new_ct.SimpleBind(ctx, args_map);
//		ct_exe->Forward(false);
//		NDArray::WaitAll();
//		NDArray ct_data = ct_exe->outputs[0];
//		args_map["ct"] = ct_data;

		//Generate ht executor
		matrix_mul(data, args_map["wo"], args_map["temp1_H"], 1, D, H);
		matrix_mul(args_map["ht"], args_map["uo"], args_map["temp2_H"], 1, H, H);
		matrix_add(args_map["temp1_H"], args_map["temp2_H"], args_map["a_ot"], H);
		matrix_add(args_map["a_ot"], args_map["bo"], args_map["a_ot"], H);
		matrix_sigmoid(args_map["a_ot"], args_map["a_ot"], H);

//		Symbol ot = dot(input, wo) + dot(ht, uo) + bo;
//		Symbol a_ot = Activation(ot, ActivationActType::sigmoid);
//		Executor* ot_exe = a_ot.SimpleBind(ctx, args_map);
//		ot_exe->Forward(false);
//		NDArray::WaitAll();

		matrix_tanh(args_map["ct"], args_map["a_ct"], H);
		matrix_mul_elementwise(args_map["a_ct"], args_map["a_ot"], args_map["ht"], H);
//		Symbol a_ct = Activation(ct, ActivationActType::tanh);
//		Executor* act_exe = a_ct.SimpleBind(ctx, args_map);
//		act_exe->Forward(false);
//		NDArray::WaitAll();
//		args_map["ht"] =  (ot_exe->outputs[0] * act_exe->outputs[0]);
//		args_map["rht"] = args_map["ht"].Reshape(Shape(H, 1));

		//Generate output executor
		matrix_mul(args_map["fc_w"], args_map["ht"], args_map["temp1_H"], V, H, 1);
		matrix_add(args_map["temp1_H"], args_map["fc_b"], args_map["temp1_H"], H);

//		Symbol rht = Symbol::Variable("rht");
//		Symbol fc = dot(fc_w, rht) +fc_b;
//		Executor* output_exe = fc.SimpleBind(ctx, args_map);
//		output_exe->Forward(false);
//		NDArray::WaitAll();
//		auto rct = output_exe->outputs[0];
//		delete ct_exe;
//		delete ot_exe;
//		delete output_exe;
		vector<float> res;
		for (int i = 0; i < V; i++)
        {
//            cout << args_map["temp1_H"][i] << endl;
			res.push_back(args_map["temp1_H"][i]);
        }
        return res;
	}

	int load_data(string fname)
	{
		char* buffer = 0;
		float* buffer_ptr = 0;
		int total_bytes = 0;
		ifstream in(fname, ios::binary);
		if (! in.is_open())
		{
			cout << fname << " file not exists!" << endl;
			return 0;
		}
		/*---------------PHASE ONE: Read file data to buffer----------------*/
		total_bytes = (4 * (D * H + H * H + H) + H * V + V) * sizeof(float);
		buffer = (char*)calloc(total_bytes, sizeof(char));
		in.read(buffer, total_bytes);
		in.close();
		buffer_ptr = (float*)buffer;

		/*---------------PHASE TWO: Initialize weights NDArray----------------*/
		//read wi data
		args_map["wi"] = buffer_ptr;
		buffer_ptr += D*H;
		//read ui data
		args_map["ui"] = buffer_ptr;
		buffer_ptr += H*H;
		//read bi data
		args_map["bi"] = buffer_ptr;
		buffer_ptr += H;

		//read wf data
		args_map["wf"] = buffer_ptr;
		buffer_ptr += D*H;
		//read uf data
		args_map["uf"] = buffer_ptr;
		buffer_ptr += H*H;
		//read bf data
		args_map["bf"] = buffer_ptr;
		buffer_ptr += H;

		//read wo data
		args_map["wo"] = buffer_ptr;
		buffer_ptr += D*H;
		//read uo data
		args_map["uo"] = buffer_ptr;
		buffer_ptr += H*H;
		//read bo data
		args_map["bo"] = buffer_ptr;
		buffer_ptr += H;

		//read wc data
		args_map["wc"] = buffer_ptr;
		buffer_ptr += D*H;
		//read uc data
		args_map["uc"] = buffer_ptr;
		buffer_ptr += H*H;
		//read bc data
		args_map["bc"] = buffer_ptr;
		buffer_ptr += H;

		//read fc weights data
		args_map["fc_w"] = buffer_ptr;
		buffer_ptr += H*V;
		//read fc bias data
		args_map["fc_b"] = buffer_ptr;

		/*---------------PHASE THREE: Initialize ht, ct and temp memory----------------*/
		data = buffer;
		args_map["ht"] = (float*)calloc(H, sizeof(float));
		args_map["ct"] = (float*)calloc(H, sizeof(float));

		args_map["temp1_H"] = (float*)calloc(H, sizeof(float));
		args_map["temp2_H"] = (float*)calloc(H, sizeof(float));
		args_map["a_ft"] = (float*)calloc(H, sizeof(float));
		args_map["a_it"] = (float*)calloc(H, sizeof(float));
		args_map["a_tmp"] = (float*)calloc(H, sizeof(float));
		args_map["a_ot"] = (float*)calloc(H, sizeof(float));
		args_map["a_ct"] = (float*)calloc(H, sizeof(float));
		return 1;
	}

private:
	int D;
	int H;
	int V;
	char* data;
	map<string, float *> args_map;
	void matrix_mul(float* a, float* b, float* c, int left, int medium, int right)
	{
		int i, j, k;
		float* temp = (float*) calloc(right, sizeof(float));
		for(i = 0; i < left; i++)
		{
			memset(temp, 0, right * sizeof(float));
			for(j = 0; j < medium; j++){
				for(k = 0; k < right; k++){
					*(temp + k) += (*(a + i * medium + j)) * (*(b + j * right + k));
				}
			}
			for(k = 0; k < right; k++){
				*(c + i * right + k) = *(temp + k);
//				printf("%f\t", *(c + i * right + k));
			}
//			printf("\n");
		}
		free(temp);
	}
	void matrix_add(float* a, float* b, float* c, int n)
	{
		for (int i= 0; i < n; i++)
			*(c + i) = (*(a + i)) + (*(b + i));
	}

	void matrix_mul_elementwise(float* a, float* b, float* c, int n)
	{
		for (int i= 0; i < n; i++)
			*(c + i) = (*(a + i)) * (*(b + i));
	}

	void matrix_sigmoid(float* a, float* b, int n)
	{
		for (int i= 0; i < n; i++)
			*(b+i) = 1 / (1 + exp(-(*(a+i))));
	}

	void matrix_tanh(float* a, float* b, int n)
	{
		for (int i= 0; i < n; i++)
			*(b+i) = (exp(*(a+i)) - exp(-(*(a+i)))) / (exp(*(a+i)) + exp(-(*(a+i))));
	}
};

#endif /* LSTM_NO_MXNET_H_ */
