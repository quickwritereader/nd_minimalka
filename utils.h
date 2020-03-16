#pragma once
#include "NdArrayMinimal.h" 
#include <LoopCoordsHelper.h>

template<size_t outter_loops = 1, size_t inner_loops = 1,typename Op, typename... Args >
void time_it(Op op, double totalFlops, Args&&... args) {
	std::vector<double> values;

	size_t n_1 = outter_loops > 1 ? outter_loops - 1 : 1;
	for (int e = 0; e < outter_loops; e++) {
		auto timeStart = std::chrono::system_clock::now();

		for (int i = 0; i < inner_loops; i++)
			op(std::forward<Args>(args)...);

		auto timeEnd = std::chrono::system_clock::now();
		auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
		values.emplace_back((double)elapsed_time / double(inner_loops));
	}

	std::sort(values.begin(), values.end());
	auto sum = std::accumulate(std::begin(values), std::end(values), 0.0);
	double avg = (double)sum / outter_loops;
	auto sum_sq = std::accumulate(std::begin(values), std::end(values), 0.0, [&avg](int sumsq, int x) { return sumsq + (x - avg) * (x - avg); });

	if (totalFlops > 0.0) {
		nd4j_printf("Median: %f us\tAvg: %f (sd: %f)\tFlops: %f Mflops\n",
			values[values.size() / 2], avg, sqrt(sum_sq / n_1),
			(totalFlops) / avg);
	}
	else {
		nd4j_printf("Median: %f us\tAvg: %f (sd: %f)\n",
			values[values.size() / 2], avg, sqrt(sum_sq / n_1));
	}
}

template<typename T>
void fill_matrice_lastC(sd::NDArray& arr, sd::NDArray* fill = nullptr, bool zero = false) {
	Nd4jLong coords[MAX_RANK] = {};
	constexpr int last_ranks = 2;
	std::random_device rd;
	std::mt19937 gen(rd());
	//for floats
	std::uniform_real_distribution<T> dis((T)0.0, (T)2.5);
	T* x = arr.bufferAsT<T>();
	Nd4jLong* shapeInfo = arr.getShapeInfo();
	Nd4jLong* strides = arr.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);

	if (rank < last_ranks) {
		return;
	}
	bool random = (fill == nullptr);
	size_t t = 1;
	const int r = rank - last_ranks;
	size_t M = bases[rank - 2];
	size_t stride_m = strides[rank - 2];
	size_t N = bases[rank - 1];
	size_t stride_n = strides[rank - 1];
	for (size_t i = 0; i < rank - last_ranks; i++) {
		t *= bases[i];
	}
	size_t offset = 0;
	if (random) {
		//first matrix is random
		//next ones copy
		if (zero) {
			for (size_t i = 0; i < t; i++) {
				//1 time
				for (size_t j = 0; j < M; j++) {
					for (size_t n_i = 0; n_i < N; n_i++) {
						x[offset + j * stride_m + n_i * stride_n] = 0;
					}
				}
				offset = sd::inc_coords(bases, strides, coords, offset, r);
			}
		}
		else {
			for (size_t i = 0; i < 1; i++) {
				//1 time
				for (size_t j = 0; j < M; j++) {
					for (size_t n_i = 0; n_i < N; n_i++) {
						x[j * stride_m + n_i * stride_n] =   dis(gen) + (T)0.5;
						//nd4j_printf("%lf  %ld  %ld %ld\n", x[j * stride_n + n_i], stride_n,j,n_i);
					}
				}
				offset = sd::inc_coords(bases, strides, coords, offset, r);
			}


			for (size_t i = 1; i < t; i++) {
				//copy_from the first
				for (size_t j = 0; j < M; j++) {
					for (size_t n_i = 0; n_i < N; n_i++) {
						x[offset + j * stride_m + n_i * stride_n] = x[j * stride_m + n_i * stride_n];
					}
				}
				offset = sd::inc_coords(bases, strides, coords, offset, r);
			}
		}
	}
	else {
		auto fill_buffer = fill->bufferAsT<T>();
		for (size_t i = 0; i < t; i++) {
			//m*n
			auto fill_stride_m = fill->stridesOf()[0];
			auto fill_strinde_n = fill->stridesOf()[1];
			for (size_t j = 0; j < M; j++) {
				for (size_t n_i = 0; n_i < N; n_i++) {
					x[offset + j * stride_m + n_i * stride_n] = fill_buffer[j * fill_stride_m + n_i * fill_strinde_n];
				}
			}
			offset = sd::inc_coords(bases, strides, coords, offset, r);
		}

	}
}



template<typename T>
bool check_eq(sd::NDArray& arr, sd::NDArray& arr2,T abs_err=(T)0.0001) {
	Nd4jLong coords[MAX_RANK] = {};
 
	Nd4jLong* shapeInfo = arr.getShapeInfo();
	Nd4jLong* shapeInfo2 = arr2.getShapeInfo();
	Nd4jLong* strides = arr.stridesOf();
	Nd4jLong* strides2 = arr.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	T *buff1 = arr.bufferAsT<T>();
	T* buff2 = arr2.bufferAsT<T>();
	if (rank != shapeInfo2[0]) {
		fprintf(stderr, "rank1 %lli rank2 %lli\n", rank, shapeInfo2[0]);
		return false;
	}
	Nd4jLong* bases = &(shapeInfo[1]);
	Nd4jLong* bases2 = &(shapeInfo2[1]);
	Nd4jLong t = 1;
	for (size_t i = 0; i < rank  ; i++) {
		t *= bases[i];
		if (bases[i] != bases2[i]) {
			fprintf(stderr, "bases index %d)  %lli vs %lli\n",i,bases[i],bases2[i]);
			return false;
		}
	}
	sd::zip_size_t offset = {};
	T max_diff =(T) 0;
			for (size_t i = 0; i < t; i++) {
				T diff = std::abs(buff1[offset.first] - buff2[offset.second]);
#if 0
				  if (diff > abs_err) {
					fprintf(stderr, "[");
					for (int i = 0; i < rank - 1; i++) {
						fprintf(stderr, "%ld, ",coords[i]);
					}
					fprintf(stderr, "%ld] : %.9g vs %.9g\n", coords[rank-1], buff1[offset.first] , buff2[offset.second]);
				  }
#endif
				max_diff = std::max(max_diff,diff  );
				offset = sd::inc_coords(bases, strides,strides2, coords, offset,rank);
			}
			fprintf(stderr, "max difference %.9g \n", max_diff);
	return max_diff>abs_err;
}