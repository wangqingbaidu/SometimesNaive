/*
 * AudioDetection.h
 *
 *  Created on: 2017年1月3日
 *      Author: zhangzhiwei
 */

#ifndef AUDIODETECTION_H_
#define AUDIODETECTION_H_
#include "LSTM_no_mxnet.h"
using namespace std;
typedef struct audiosegment
{
	int start;
	int end;
}audio_segments;

class AudioDetection {
public:
	AudioDetection(){}
	AudioDetection(int D, int H, int V, const char* path): lstm(D,H,V,path){
		this->D = D;
		this->H = H;
		this->V = V;
	}
	virtual ~AudioDetection(){}

	void getDetection(vector<vector<float>>& res, vector<short>& wav, vector<vector<float>>& feature)
	{
		int min_segment = ceil(min_duration / segment_duration);
		int min_split = ceil(start_duration / segment_duration);
		int energy_npoints = fs * enegy_duration / 2;
		int npoints = fs * segment_duration;
		vector<bool> labels;

		//Reset LSTM state
		lstm.reset_state();
		//Get label
		for(vector<vector<float>>::iterator it = feature.begin(); it != feature.end(); it++)
		{
			if(it->size() != D)
			{
				cout << "Error feature size" << endl;
				exit(0);
			}
			vector<float> a = lstm.get_output(it->data());
			if (a.at(0) < a.at(1))
				labels.push_back(false);
			else
				labels.push_back(true);
		}

		//Post process
		vector<audio_segments> seg;
		seg = pure_boundaries(labels, wav.data(), wav.size(), min_segment, min_split, energy_npoints, npoints);

		for (int i = 0; i < seg.size(); i++)
		{
	//		cout << seg.at(i).start << ' ' << seg.at(i).end << endl;
			vector<float> s;
			s.push_back(seg.at(i).start * segment_duration);
			s.push_back(seg.at(i).end * segment_duration);
			res.push_back(s);
		}
	}
private:
	int D; int H; int V;
	LSTM lstm;
	int fs = 16000;
	float min_duration = 0.35;
	float segment_duration = 0.02;
	float start_duration = 1;
	float enegy_duration = 0.05;

	void label_to_boundary(vector<audio_segments> &res, vector<bool> &labels, int min_segment)
	{
		int start_idx = -1;
		int i;
		bool change = false;
		for (i = 0; i < labels.size(); i++)
		{
			if (labels.at(i) && !change)
			{
				change = true;
				start_idx = i;
			}
			if (!labels.at(i) && change)
			{
				change = false;
				if ((i - start_idx) >= min_segment)
					res.push_back(audio_segments{start_idx, i});
				start_idx = -1;
			}
		}
		if (start_idx != -1)
		{
			if ((i - start_idx)  >= min_segment)
				res.push_back(audio_segments{start_idx, i});
		}
	}

	//Calculate energy of given wav.
	//stride: points of step
	//npoints: the number of points in sample
	void calc_energy(vector<double> &out, short* wav, int wav_length, int stride, int npoints,
			bool log_domain, float smoothing = 1.0)
	{
		int start_idx = 0;
		while ((start_idx + npoints) < wav_length)
		{
			double mean = 0;
			for (int i = 0; i < npoints; i++){
				mean += pow(wav[start_idx + i], 2) + smoothing;
			}

			mean /= npoints;
			if (log_domain)
				out.push_back(log(sqrt(mean)));
			else
				out.push_back(sqrt(mean));
			start_idx += stride;
		}
	}

	double get_energy_threshold(vector<double> &energy, int s, int e, float percentage = 0.1)
	{
		vector<double> energy_sorted;
		double max_e = 0, min_e = 0xfffffff;
		for (int i = s; i < e; i++)
		{
			if (max_e < energy.at(i))
				max_e = energy.at(i);
			if (min_e > energy.at(i))
				min_e = energy.at(i);
		}

		sort(energy_sorted.begin(), energy_sorted.end());
		int idx = int(energy_sorted.size() * percentage);
		return min_e + (max_e - min_e) * percentage;
	}

	vector<audio_segments> get_split(vector<double> energy, int s, int e, int min_segment,
			float percentage=0.1)
	{
		vector<double> energy_sorted;
		for (int i = s; i < e; i++)
			energy_sorted.push_back(energy.at(i));
		sort(energy_sorted.begin(), energy_sorted.end());
		//Get mean
		double sum = 0;
		for(vector<double>::iterator it = energy_sorted.begin(); it != energy_sorted.end(); it++)
			sum += *it;
		//cout << sum << endl;
		double mean =  sum / energy_sorted.size(); //mean
		if (mean != 0)
		{
			//Get std
			double accum  = 0.0;
			for(vector<double>::iterator it = energy_sorted.begin(); it != energy_sorted.end(); it++)
				accum += (*it - mean) * (*it - mean);
			double std = sqrt(accum/energy_sorted.size()); //方差

			percentage *= max(0.0, log(std / mean * 10 + 1.0e-8) + 1);
		}
		cout << percentage << endl;
		int idx = int(energy_sorted.size() * percentage);
		double threshold = energy_sorted.at(idx);
		int i, start_idx = -1;;
		bool change = false;
		vector<audio_segments> split, res;

		for (i = s; i < e; i++)
		{
			if ((energy.at(i) >= threshold) && !change)
			{
				change = true;
				start_idx = i;
			}
			if ((energy.at(i) < threshold) && change)
			{
				split.push_back(audio_segments{start_idx, i});
				change = false;
				start_idx = -1;
			}
		}
		if (start_idx != -1)
		{
			split.push_back(audio_segments{start_idx, i});
			start_idx = -1;
		}

		for (i = 0; i < split.size(); i++)
		{
			if (start_idx == -1)
				start_idx = split.at(i).start;
			if ((split.at(i).end - start_idx) < min_segment)
				continue;
			else
			{
				res.push_back(audio_segments{start_idx, split.at(i).end});
				start_idx = -1;
			}
		}
		return res;
	}

	vector<double> smooth_moving_average(vector<double> &signal, int window_len = 11)
	{
		if (window_len < 3)
		{
			cout << "Window length must larger than 3" << endl;
			return signal;
		}

		if (signal.size() < window_len)
		{
			cout << "Signal size must be larger than windowLen!" << endl;
			exit(0);
		}

		int start_len = window_len / 2;
		int end_len = window_len % 2 ? window_len / 2 : window_len / 2 - 1;
		vector<double> start;
		vector<double> end;
		for (int i = 0; i < start_len; i++)
			start.push_back(2 * signal.at(0) - signal.at(start_len - i - 1));
		for (int i = 0; i < end_len; i++)
			end.push_back(2 * signal.at(signal.size() - 1) - *(signal.end() - i - 1));
		start.insert(start.end(), signal.begin(), signal.end());
		start.insert(start.end(), end.begin(), end.end());

	//	for(vector<double>::iterator it = start.begin(); it != start.end(); it++)
	//		cout << *it << endl;

		double sum = 0;
		vector<double> res;
		for(int i = 0; i < window_len; i++)
			sum += start.at(i);

		res.push_back(sum / window_len);
		for (int i = 1; i <= start.size() - window_len; i++)
		{
			sum = sum + start.at(window_len + i - 1) - start.at(i - 1);
			res.push_back(sum / window_len);
		}
		return res;
	}

	//min_segment: number of each fragment.
	//min_split: split segment which contains more than min_split points
	//energy_npoints: for energy_log
	//npoints: points of energy, per fragment.
	vector<audio_segments> pure_boundaries(vector<bool> &labels, short* wav, int wav_length,
			int min_segment, int min_split, int energy_npoints, int npoints)
	{
		//Get boundaries
		vector<audio_segments> segments;
		label_to_boundary(segments, labels, min_segment);

	//	for (int i = 0; i < segments.size(); i++)
	//		cout << segments.at(i).start << ' ' << segments.at(i).end << endl;
	//	cout << endl;

		//Post process
		vector<audio_segments> new_boundaries;
		vector<double> energy_log;
		vector<double> energy;
		calc_energy(energy_log, wav, wav_length, energy_npoints, energy_npoints * 2, true);
		calc_energy(energy, wav, wav_length, npoints, npoints, false);
		vector<double> smooth_energy_log = smooth_moving_average(energy_log);
		vector<double> smooth_energy = smooth_moving_average(energy);

		for (int i = 0; i < segments.size(); i++)
		{
			if ((segments.at(i).end - segments.at(i).start) <= min_split)
			{
				new_boundaries.push_back(segments.at(i));
				continue;
			}
			else
			{
				//Todo: add min split algorithm
				int log_min_idx = segments.at(i).start * npoints / energy_npoints;
				int log_max_idx = min(segments.at(i).end * npoints / energy_npoints, int(smooth_energy_log.size()));
				vector<audio_segments> split_log;
				split_log = get_split(smooth_energy_log, log_min_idx, log_max_idx, min_segment * npoints / energy_npoints);
				for (vector<audio_segments>::iterator it = split_log.begin(); it != split_log.end(); ++it)
				{
					int start_idx = it->start * energy_npoints / npoints;
					int end_idx = it->end * energy_npoints / npoints;
					double threshold = get_energy_threshold(smooth_energy, start_idx, end_idx);
					while(smooth_energy.at(start_idx) < threshold)
						start_idx++;
					while(smooth_energy.at(end_idx) < threshold)
						end_idx--;
					new_boundaries.push_back(audio_segments{start_idx, end_idx + 1});
				}
			}
		}
		return new_boundaries;
	}
};

#endif /* AUDIODETECTION_H_ */
