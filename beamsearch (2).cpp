#include <fstream>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>
#include <numeric>
#include "NdArrayMinimal.h"
#include "LoopCoordsHelper.h"
#include "Threads.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
using namespace sd;

std::vector<float> load_file(const std::string& filename)
{
	std::fstream file_stream(filename);
	std::vector<float> data;
	data.reserve(8 * 1024);
	int col_size = 0;
	if (file_stream.is_open())
	{
		int count_y = 0;
		bool counted_col = false;
		std::string numb;
		while (std::getline(file_stream, numb, ';'))
		{
			const auto pos = numb.find('\n');
			if (pos != std::string::npos)
			{
				numb = numb.erase(pos, 1);
				if (!counted_col) col_size = count_y;
				counted_col = true;
			}
			++count_y;
			if (!numb.empty()) data.push_back(std::stof(numb));
		}

		file_stream.close();
	}
	else std::cout << "Unable to open file";
	std::cout << data.size() << std::endl;
	std::cout << data.data() << std::endl;
	std::cout << data.size() / col_size << " " << col_size << std::endl;
	return data;
}

template <int constRank = 2, bool IsFortran = false>
NDArray Word2LabelSeq(std::string word, const std::string& classes)
{
	constexpr int skip = 0;
	auto dim = std::vector<Nd4jLong>(constRank, 1); //calls ctor(count, val)
	dim[constRank - 1] = word.size();
	auto res = NDArrayFactory::create<int>(IsFortran ? 'c' : 'f', dim);
	const auto x = static_cast<int*>(res.buffer());
	CoordsState<constRank - 1> cst;
	int wi = 0;
	size_t offset = sd::init_coords<constRank, skip, IsFortran>(cst, 0, dim.data(), res.stridesOf());
	const size_t loop_count = 1 * word.size();
	for (size_t i = 0; i < loop_count; i++)
	{
		x[offset] = classes.find(word[wi++]);
		offset = sd::inc_coords<constRank, skip, IsFortran>(cst, offset);
	}
	return res;
}

template <typename T, int constRank = 2>
void softmax(NDArray& input, bool makeLog = false)
{
	constexpr int skip = 0;
	CoordsState<constRank> cst;
	const auto x_shapeInfo = input.shapeInfo();
	T* x = input.bufferAsT<T>();
	auto bases = &(x_shapeInfo[1]);
	auto x_strides = &(x_shapeInfo[constRank + 1]);
	size_t offset = sd::init_coords<constRank, skip, false>(cst, 0, bases, x_strides);
	size_t loop_count = 1;
	for (Nd4jLong i = 0; i < constRank - 1; i++)
	{
		loop_count *= bases[i];
	}
	std::cout << loop_count << " , " << LAST_NUM(cst, 0) << " , " << STRIDE(cst, 0);
	std::cout << " , " << LAST_NUM(cst, 1) << " , " << STRIDE(cst, 1) << std::endl;
	auto el_stride = STRIDE(cst, constRank - 1);
	auto last_index = LAST_NUM(cst, constRank - 1);
	T min_val = std::numeric_limits<T>::min();
	for (auto i = 0UL; i < loop_count; i++)
	{
		T* lx = x + offset;
		auto max_val = lx[0];
		for (Nd4jLong j = 0; j <= last_index; j++)
		{
			max_val = lx[j * el_stride] > max_val ? lx[j * el_stride] : max_val;
		}

		T denom_sum = 0;
		for (Nd4jLong j = 0; j <= last_index; j++)
		{
			T val = std::exp(lx[j * el_stride] - max_val);
			denom_sum += val;
			lx[j * el_stride] = val;
		}
		if (!makeLog)
		{
			for (auto j = 0; j <= last_index; j++)
			{
				lx[j * el_stride] = lx[j * el_stride] / denom_sum;
				if (lx[j * el_stride] < min_val) lx[j * el_stride] = min_val;
			}
		}
		else
		{
			for (Nd4jLong j = 0; j <= last_index; j++)
			{
				lx[j * el_stride] = std::log(lx[j * el_stride]) - std::log(denom_sum);
			}
		}
		offset = sd::inc_coords<constRank, skip, false>(cst, offset);
	}
}


template <typename T>
constexpr T negative_infinity()
{
	return -std::numeric_limits<T>::infinity();
}

template <typename T>
T log_sum_exp(T arg1, T arg2, T arg3)
{
	auto c_max = std::max(arg1, arg2);
	c_max = std::max(c_max, arg3);
	if (negative_infinity<T>() == c_max)
	{
		c_max = 0;
	}
	return std::log(std::exp(arg1 - c_max) + std::exp(arg2 - c_max) + std::exp(arg3 - c_max)) + c_max;
}

template <typename T>
T local_log(T x)
{
	if (x > 0)
	{
		return (std::log(x));
	}
	return (negative_infinity<T>());
}

template <typename T>
T log_sum_exp(T x1, T x2)
{
	//substituting this : std::log(std::exp(arg1 - cMax) + std::exp(arg2 - cMax)) + cMax
	//if arg1==cMax : std::log(1 + std::exp(arg2 - cMax)) + cMax
	if (x1 >= x2)
	{
		//x1 is max
		return (x1 + local_log(1 + std::exp(x2 - x1)));
	}
	//x2 is max
	return (x2 + local_log(1 + std::exp(x1 - x2)));
}

template <typename T>
struct BeamProb
{
	T total = negative_infinity<T>();
	T non_blank = negative_infinity<T>();
	T blank = negative_infinity<T>(); //log(1)
};


template <typename T, typename T2 = void>
struct DefaultInvalid
{
	static constexpr T value = T();
};


template <typename T>
struct DefaultInvalid<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
	static constexpr T value = static_cast<T>(-1);
};


template <typename T>
struct SequenceNode;
template <typename T>
struct SequenceContainer;


template <typename T>
struct SequenceNode
{
	//intrusive double links
	SequenceNode<T>* prev = nullptr;
	SequenceNode<T>* next = nullptr;

	//sequence prefix/parent
	SequenceNode<T>* prefix = nullptr;

	T value = DefaultInvalid<T>::value;

	int state = 0;

	void markAsFullyExtended()
	{
		state |= 1;
	}

	void increaseRef()
	{
		state |= 2;
		//std::cout << "markas " << (long long)this << " "<<std::endl;
		//print_seq1(this);
	}

	void decreaseRef()
	{
		state = state & (-2);
	}
	bool safeToDelete() const
	{
		return state & 2;
	}
	bool isFullyExtended() const { return state & 1; }
};


template <typename T>
class SequenceContainer
{
public:
	SequenceContainer() : current_(new SequenceNode<T>()), count_(1)
	{
	}

	SequenceContainer(const SequenceContainer& s) = delete;

	SequenceContainer(SequenceContainer&& other) noexcept
	{
		this->current_ = other.current_;
		other.current_ = nullptr;
	}

	SequenceContainer& operator=(const SequenceContainer& other) = delete;

	SequenceContainer& operator=(SequenceContainer&& other) noexcept
	{
		if (this != other)
		{
			clear();
			this->current_ = other.current_;
			this->count_ = other.count_;
			other.current_ = nullptr;
			other.count_ = 0;
		}
		return *this;
	}

	SequenceNode<T>* getEmptyPath()
	{
		return current_;
	}

	SequenceNode<T>* extendPath(SequenceNode<T>* prefix, T value)
	{
		auto new_node = new SequenceNode<T>();

		new_node->value = value;
		new_node->prefix = prefix;


		//add in the holder
		new_node->next = nullptr;
		new_node->prev = current_;
		/*std::cout << "add " << (long long)new_node << std::endl;
		print_seq1(new_node);*/
		if (current_) current_->next = new_node;

		current_ = new_node;
		count_++;
		return new_node;
	}

	void remove(SequenceNode<T>* seq)
	{
		if (seq == nullptr) return;
		if (seq->isFullyExtended()) return;
		if (seq->isShared())
		{
			seq->clearBeingShared();
			return;
		}

		SequenceNode<T>* previous = seq->prev;
		SequenceNode<T>* next = seq->next;
		if (previous) previous->next = next;
		if (next) next->prev = previous;

		if (current_ == seq)
		{
			current_ = previous;
		}
		//std::cout << "remove " << (long long)seq << " " << std::endl;
		//print_seq1(seq);
		delete seq;
		count_--;
	}

	static std::vector<T> getSequence(SequenceNode<T>* seq, size_t reserve_size = 1024)
	{
		std::vector<T> ret;
		ret.reserve(reserve_size);
		SequenceNode<T>* backtrack = seq;
		while (backtrack)
		{
			ret.push_back(backtrack->value);
			backtrack = backtrack->prefix;
		}
		if (ret.size() > 1)
		{
			//remove last default node
			ret.pop_back();
			//reverse
			std::reverse(std::begin(ret), std::end(ret));
			return ret;
		}
		return {};
	}

	void clear()
	{
		//destruct all nodes
		SequenceNode<T>* del = current_;
		//int i = 0;
		while (del)
		{
			//++i;
			SequenceNode<T>* temp = del->prev;
			delete del;
			del = temp;
		}
		current_ = nullptr;
		//assert(count_==i);
	}

	~SequenceContainer()
	{
		clear();
	}

private:
	SequenceNode<T>* current_ = nullptr;

	int count_ = 0;
};

void print_seq1(SequenceNode<int>* parent)
{
	//const auto lf = std::to_string(parent->isFullyExtended());
	//const auto v = std::string("__leaf");
	//if (parent->isFullyExtended())
	//	std::cout << "{" << v << "__: ";
	//else
	//	std::cout << "{__branch " << lf << "__: ";
	std::cout << "(";
	for (auto x : SequenceContainer<int>::getSequence(parent))
	{
		std::cout << x << ", ";
	}
	std::cout << ")" << std::endl;
}

void print_seq1(SequenceNode<int>* parent, int value)
{
	std::cout << "{";
	for (auto x : SequenceContainer<int>::getSequence(parent))
	{
		std::cout << x << " , ";
	}
	std::cout << value << " , ";
	std::cout << "}" << std::endl;
}

template <typename T, typename U>
struct BeamEntry
{
	SequenceNode<U>* sequence{};
	BeamProb<T> prob;
};

enum class OverlapDirection : int32_t
{
	NONE = 0,
	PARENT = 1,
	CHILD = 2
};

static OverlapDirection operator|(OverlapDirection lhs, OverlapDirection rhs)
{
	return static_cast<OverlapDirection>(
		static_cast<std::underlying_type<OverlapDirection>::type>(lhs) |
		static_cast<std::underlying_type<OverlapDirection>::type>(rhs)
		);
}

static OverlapDirection operator&(OverlapDirection lhs, OverlapDirection rhs)
{
	return static_cast<OverlapDirection>(
		static_cast<std::underlying_type<OverlapDirection>::type>(lhs) &
		static_cast<std::underlying_type<OverlapDirection>::type>(rhs)
		);
}

template <typename T, typename U>
struct BeamEntryEx
{
	BeamEntry<T, U> entry;
	//for speeding up map. we will only look for possible overlap cases suing this flag
	OverlapDirection overlap_direction = OverlapDirection::NONE;
};


template <typename T, typename U>
bool compare_beam_prob(const BeamEntry<T, U>& i1, const BeamEntry<T, U>& i2)
{
	return (i1.prob.total > i2.prob.total);
}



constexpr uint32_t bits_required(const uint32_t n)
{
	return n <= 1 ? 0 : 1 + bits_required((n + 1) / 2);
}

struct PairHash
{
	template <class T1, class T2>
	std::size_t operator()(const std::pair<T1*, T2>& p) const noexcept
	{
		//return hash_combine(p.first, p.second);
		//first is mostly aligned pointer 
		const auto tmp = (reinterpret_cast<uint64_t>(p.first) >> bits_required(sizeof(T1))) ^ (static_cast<uint64_t>(p.second)
			<< 16);

		return std::hash<uint64_t>()(tmp);
	}
};

template <typename T, typename U>
T pr(const int c, const BeamProb<T>& beam_prob, const SequenceNode<U>* seq, const T prob)
{
	return seq->value == c ? beam_prob.blank + prob : beam_prob.total + prob;
}


//choose ptr[index*element_stride] 
template <bool HasStride, typename Type>
std::enable_if_t<HasStride == true, Type&>
element(Type* ptr, int index, uint64_t element_stride)
{
	return ptr[index * element_stride];
}

//choose ptr[index] assuming element_stride is 1
template <bool HasStride, typename Type>
std::enable_if_t<HasStride == false, Type&>
element(Type* ptr, int index, uint64_t element_stride)
{
	return ptr[index];
}

//#define LOGIT_SOFTMAX_NORMALIZATION 1

template <bool HasElementStride, typename Type, typename IndexType>
Type softmax_normalization_term(const Type* log_p, const uint64_t len_c, const uint64_t element_stride)
{
	Type max_p;
	for (auto c = 0; c < len_c; ++c) {
		max_p = std::max(max_p, element<HasElementStride>(log_p, c, element_stride));
	}
	// Get normalization term of softmax: log(sum(exp(logit[j]-max_p))).
	Type logsumexp = Type(0.0);
	for (auto c = 0; c < len_c; ++c) {
		logsumexp += std::exp(element<HasElementStride>(log_p, c, element_stride) - max_p);
	}
	logsumexp = std::log(logsumexp);
	return max_p + logsumexp;
}

template<bool HasElementStride = false, typename Type, typename IndexType>
void inner_beam_search(const Type* log_p, const uint64_t inc_p, IndexType* result_sequence, const uint64_t inc_res_seq, const uint64_t max_len_t, Type* result_prob, uint64_t len_t, const uint64_t len_c, const int blank_index, int beam_width, int nbest_len, const uint64_t element_stride = 1L)
{
	using PairParentChild = std::pair<SequenceNode<IndexType>*, IndexType>;
	using BeamEntryType = BeamEntry<Type, IndexType>;
	using BeamEntryTypeEx = BeamEntryEx<Type, IndexType>;
	PairParentChild entry_point;


	if (beam_width < 1) beam_width = 1;
	if (nbest_len > beam_width) nbest_len = beam_width;
	//if len_t is greater than max_len_t truncate it
	len_t = len_t > max_len_t ? max_len_t : len_t;

	SequenceContainer<IndexType> sequence_container;
	BeamEntryType empty;
	empty.prob.blank = 0;
	empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.non_blank);
	empty.sequence = sequence_container.getLast();

	//vectors: we will use it as array, here
	std::vector<BeamEntryTypeEx> last_beams;
	std::vector<BeamEntryType> next_beams;
	last_beams.resize(beam_width);
	//as we skip blank indexes the count is beam_width * len_c 
	next_beams.resize(beam_width * len_c);
	last_beams[0] = { empty, OverlapDirection::NONE };
	auto last_beam_size = 1;
	//combination(2)(beam_width + 1) = (beam_width-1)*beam_width/2!
	std::unordered_map<PairParentChild, int, PairHash> map_index;

	map_index.reserve((beam_width - 1) * beam_width / 2);

	std::vector<bool> overlap_classes;
	overlap_classes.resize(len_c);

	for (uint64_t t = 0; t < len_t; t++)
	{
		auto next_beam_size = 0;
		//std::cout << "---" << t << "---" << std::endl;
		for (auto j = 0; j < last_beam_size; j++)
		{
			SequenceNode<IndexType>* seq = last_beams[j].entry.sequence;
			auto& cur_prob = last_beams[j].entry.prob;
			const OverlapDirection overlap_dir_parent = last_beams[j].overlap_direction & OverlapDirection::PARENT;
			const OverlapDirection overlap_dir_child = last_beams[j].overlap_direction & OverlapDirection::CHILD;
			//if len(seq) > 0 then
			const auto log_p_blank = element<HasElementStride>(log_p, blank_index, element_stride);

			//log_p[seq->value] 
			auto non_blank_prob = seq->value != -1 ? (element<HasElementStride>(log_p, seq->value, element_stride) + cur_prob.non_blank) : negative_infinity<Type>();

			const Type blank_prob = log_p_blank + cur_prob.total;



			auto found = std::end(map_index);

			if (overlap_dir_child == OverlapDirection::CHILD)
			{
				//check entry
				seq->markAsShared();
				entry_point = std::make_pair(seq->prefix, seq->value);
				found = map_index.find(entry_point);
			}

			if (found == std::end(map_index))
			{
				BeamEntryType entry;
				entry.sequence = seq;
				entry.prob.blank = blank_prob;
				entry.prob.non_blank = non_blank_prob;
				entry.prob.total = log_sum_exp(blank_prob, non_blank_prob);
				next_beams[next_beam_size] = entry;
				//map if its overlapped one. in this case just being child is enough
				if (overlap_dir_child == OverlapDirection::CHILD)
				{
					map_index[entry_point] = next_beam_size;
				}
				++next_beam_size;
			}
			else
			{
				//note: here we took as ref &
				auto& entry_prob = next_beams[found->second].prob;
				entry_prob.blank = log_sum_exp(entry_prob.blank, blank_prob);
				entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
				entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.non_blank);
			}
			//check to see if it was extended previously

			for (int c = 0; c < len_c; c++)
			{
				if (c == blank_index) continue;

				const auto prob = element<HasElementStride>(log_p, c, element_stride);//log_p[c];

				non_blank_prob = pr(c, cur_prob, seq, prob);

				found = std::end(map_index);
				//extend by new character 

				if (overlap_dir_parent == OverlapDirection::PARENT)
				{
					entry_point = std::make_pair(seq, c);
					found = map_index.find(entry_point);
				}
				if (found == std::end(map_index))
				{
					BeamEntryType entry;
					SequenceNode<IndexType>* new_sequence;
					if (overlap_dir_parent == OverlapDirection::PARENT && seq->isFullyExtended() && overlap_classes[c])
					{
						bool find = false;
						//try to find sequence in time frame t beams
						for (int tb = 0; tb < last_beam_size; tb++)
						{
							auto ss = last_beams[tb].entry.sequence;
							if (ss->prefix == seq && ss->value == c)
							{
								new_sequence = ss; //already extended one for this time frame;
								ss->markAsShared();
								find = true;
								break;
							}
						}
						if (!find) new_sequence = sequence_container.add(seq, c);
					}
					else {
						new_sequence = sequence_container.add(seq, c);
					}
					entry.prob.non_blank = non_blank_prob;
					entry.prob.total = non_blank_prob;
					entry.sequence = new_sequence;
					next_beams[next_beam_size] = entry;

					//we will map only ones that can overlap
					if (overlap_dir_parent == OverlapDirection::PARENT && overlap_classes[c]) {
						map_index[entry_point] = next_beam_size;
					}
					++next_beam_size;
				}
				else
				{
					auto& entry_prob = next_beams[found->second].prob;
					entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
					entry_prob.total = log_sum_exp(entry_prob.total, non_blank_prob);
				}
			} //iteration over classes

			//mark it as extended
			seq->markAsExtended();

		} //iteration over  beams

		log_p += inc_p;

#if 0
		std::cout << "---" << t << "---" << std::endl;
		//print
		for (int j = 0; j < next_beam_size; j++)
		{
			std::cout << " in container " << (long long)next_beams[j].sequence << std::endl;
			print_seq1(next_beams[j].sequence);
			//std::cout << std::setprecision(17) << next_beams[j].prob.total << std::endl;
		}
		std::cout << "\n\n\n" << std::endl;
#endif
		last_beam_size = std::min(next_beam_size, beam_width);

		//sort next beams to get candidates
		std::partial_sort(std::begin(next_beams),
			std::begin(next_beams) + last_beam_size,
			std::begin(next_beams) + next_beam_size, compare_beam_prob<Type, IndexType>);
		// std::sort(std::begin(next_beams), std::begin(next_beams) + next_beam_size, compare_beam_prob<Type, IndexType>);


		if (t < len_t)
		{
			//copy top beams
			for (int j = 0; j < last_beam_size; j++)
			{
				last_beams[j].entry = next_beams[j];
				last_beams[j].overlap_direction = OverlapDirection::NONE;
			}

			//reset 
			map_index.clear();

			//delete sequences from the sequence_holder to decrease memory
			for (auto j = beam_width; j < next_beam_size; j++)
			{
				sequence_container.remove(next_beams[j].sequence);
			}

			//assume there is not any class that can be overlapped
			for (auto j = 0; j < len_c; j++)
			{
				overlap_classes[j] = false;
			}

			//check overlap_direction
			for (auto j = 0; j < last_beam_size; j++)
			{
				if (last_beams[j].entry.sequence->isFullyExtended())
				{
					auto overlaps = false;

					for (int k = 0; k < last_beam_size; k++)
					{
						if (last_beams[k].entry.sequence->prefix == last_beams[j].entry.sequence)
						{
							last_beams[k].overlap_direction = last_beams[k].overlap_direction | OverlapDirection::CHILD;
							overlap_classes[last_beams[k].entry.sequence->value] = true;
							//last_beams[k].entry.sequence->SetLeafVal(j);
							overlaps = true;
						}
					}
					if (overlaps)
					{
						last_beams[j].overlap_direction = last_beams[j].overlap_direction | OverlapDirection::PARENT;
					}
				}
			} //overlap_direction identified to speed up lookUp
		}
	}

	//store nbest results
	if (nbest_len <= last_beam_size) {
		for (int j = 0; j < nbest_len; j++)
		{
			auto top = next_beams[j];
			auto result_vector = SequenceContainer<IndexType>::getNodeSequence(top.sequence, len_t);
			const auto seq_size = result_vector.size();

			result_prob[j] = top.prob.total;
			//copy sequence
			for (auto s = 0; s < seq_size; s++)
			{
				result_sequence[s] = result_vector[s];
			}
			//mark end of sequence with blank if space left
			if (seq_size < max_len_t)
			{
				result_sequence[seq_size] = blank_index;
			}
			result_sequence += inc_res_seq;

		}
	}
	else
	{
		for (int j = 0; j < nbest_len; j++)
		{
			result_prob[j] = negative_infinity<Type>();
			result_sequence[j * inc_res_seq] = blank_index;
		}
	}
	return;
}

/**
 * inputs	3-D float Tensor, size [max_time, batch_size, num_classes]. The logit.
sequence_length	1-D int32 vector containing sequence lengths, having size [batch_size].
beam_width	An int scalar >= 0 (beam search beam width).
top_paths	An int scalar >= 0, <= beam_width (controls output size).
 */

 /**
  *
  */
template<typename Type, typename IndexType = int>
void
beamSearch(const NDArray& logit, const NDArray& sequence_length, NDArray& result_sequences, NDArray& result_probs, int blank_index, int beam_width = 25, int nbest_len = 1)
{

	const auto shapes = logit.shapeOf();
	const auto strides = logit.stridesOf();
	const auto rank = logit.rankOf();

	IndexType* len_t_ptr = nullptr;
	uint64_t element_stride_t = 1;

	//checks before
	if (rank < 2) return;
	auto batch_len = rank > 2 ? shapes[0] : 1;
	auto max_len_t = shapes[rank - 2];
	auto len_c = shapes[rank - 1];

	if (len_c < 1 || max_len_t < 1) return;
	//defaulting blankIndex to the last class if its incorrect or -1
	if (blank_index > len_c || blank_index < 0) blank_index = static_cast<int>(len_c) - 1;
	if (sequence_length.rankOf() == 1 && sequence_length.shapeOf()[0] == batch_len)
	{
		len_t_ptr = sequence_length.bufferAsT<IndexType>();
		element_stride_t = sequence_length.stridesOf()[0];
	}

	//strides
	auto batch_stride = rank > 2 ? strides[0] : 0;
	auto inc_p = strides[rank - 2];
	auto element_stride = logit.stridesOf()[rank - 1];

	auto logits_ptr = logit.bufferAsT<Type>();

	//result_probs should be [batch_len, nbest_len]
	if (result_probs.ews() == 1 && result_probs.rankOf() == 2 && result_probs.shapeOf()[0] == batch_len && result_probs.shapeOf()[1] == nbest_len)
		//result sequence should be [batch_len, nbest_len,  max_len_t]
		assert(result_sequences.ews() == 1 && result_sequences.rankOf() == 3 && result_sequences.shapeOf()[0] == batch_len && result_sequences.shapeOf()[1] == nbest_len && result_sequences.shapeOf()[2] == max_len_t);

	auto result_seq_ptr = result_sequences.bufferAsT<IndexType>();
	auto result_probs_ptr = result_probs.bufferAsT<Type>();

	const auto  batch_stride_res = result_sequences.stridesOf()[0];
	const auto  inc_res = result_sequences.stridesOf()[1];
	const auto  batch_stride_res_prob = result_probs.stridesOf()[0];

	auto func = [max_len_t, len_c, batch_stride, inc_p, element_stride, element_stride_t, logits_ptr, len_t_ptr, blank_index, beam_width,
		nbest_len, result_seq_ptr, result_probs_ptr, batch_stride_res, inc_res, batch_stride_res_prob]
		(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void
	{

		auto ptr = logits_ptr + start * batch_stride;

		if (element_stride == 1)
		{
			//choose ews one
			for (auto b = start; b < stop; b += increment)
			{
				auto res_prob_ptr = &(result_probs_ptr[b * batch_stride_res_prob]);
				auto res_seq_ptr = &(result_seq_ptr[b * batch_stride_res]);

				auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
				inner_beam_search<false, Type, IndexType>(ptr, inc_p, res_seq_ptr, inc_res, max_len_t, res_prob_ptr, len_t, len_c, blank_index, beam_width, nbest_len);

				ptr += batch_stride;

			}
		}
		else
		{
			// element with stride case 
			for (auto b = start; b < stop; b += increment)
			{
				auto res_prob_ptr = &(result_probs_ptr[b * batch_stride_res_prob]);
				auto res_seq_ptr = &(result_seq_ptr[b * batch_stride_res]);

				auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
				inner_beam_search<false, Type, IndexType>(ptr, inc_p, res_seq_ptr, inc_res, max_len_t, res_prob_ptr, len_t, len_c, blank_index, beam_width, nbest_len, element_stride);

				ptr += batch_stride;
			}
		}
	};
	samediff::Threads::parallel_for(func, 0, batch_len, 1);
	return;
}


template <typename Op >
void testb(Op op, const NDArray& logInput, const NDArray& seq_len, const std::string& classes, int w = 3, int nbest = 1)
{
	int blankInd = static_cast<int>(logInput.shapeOf()[logInput.rankOf() - 1] - 1L);
	const auto batch_len = logInput.rankOf() > 2 ? logInput.shapeOf()[0] : 1;
	const auto max_len_t = logInput.shapeOf()[logInput.rankOf() - 2];
	std::cout << "OP: blankInd " << blankInd << ", " << w << std::endl;
	auto result_prob = NDArrayFactory::create<float>('c', { batch_len, nbest });
	auto result_seq = NDArrayFactory::create<int>('c', { batch_len , nbest, max_len_t });

	op(logInput, seq_len, result_seq, result_prob, blankInd, w, nbest);

	result_seq.printIndexedBuffer("result seq");
	result_prob.printIndexedBuffer("result prob");

	auto res_ptr = result_seq.bufferAsT<int>();
	const auto inc = result_seq.stridesOf()[1];
	//print one of seq as lchars
	for (int j = 0; j < nbest; j++) {
		for (int i = 0; i < max_len_t; i++) {
			if (res_ptr[i] == blankInd)
			{
				break; //as seq length was shorter
			}
			std::cout << classes[res_ptr[i]] << ",";
		}
		std::cout << std::endl;
		res_ptr += inc;
	}
}


constexpr int global_beam_width = 25;

extern "C" float test_new_improved()
{
	static NDArray logInput, seq_length, result_prob, result_seq;
	static int t = 0;
	const std::string classes = " !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'";
	if (t == 0)
	{
		const std::vector<float> data = load_file("rnn.csv");
		logInput = NDArrayFactory::create<float>('c', { 1, 100, 80 }, data);
		seq_length = NDArrayFactory::create<int>('c', { 1 }, { 100 });
		result_prob = NDArrayFactory::create<float>('c', { 1, 1 });
		result_seq = NDArrayFactory::create<int>('c', { 1 , 1, 100 });
		softmax<float, 3>(logInput, true);
		t = 1;
	}


	beamSearch<float, int>(logInput, seq_length, result_seq, result_prob, logInput.shapeOf()[1] - 1, global_beam_width, 1);
	return result_prob.bufferAsT<float>()[0];
}


void test123()
{
	constexpr int CLASS_LEN = 5;
	constexpr int BATCH_LEN = 1;
	constexpr int MIN_FRAME_LEN = 4;
	constexpr int MAX_FRAME_LEN = 6;
	constexpr int NBEST_LEN = 5;
	constexpr int BEAM_WIDTH = 3;
	constexpr int BLANK_INDEX = CLASS_LEN - 1;

	auto logits = NDArrayFactory::create<float>('c', { BATCH_LEN, MAX_FRAME_LEN, CLASS_LEN },
		{
			-1.52900087f,-1.7423916f,-1.79369985f,-1.68980741f,-1.35771429f,
-2.08261997f,-1.65483307f,-1.31878488f,-1.38940393f,-1.78624192f,
-1.83125744f,-1.28989651f,-1.86882736f,-1.51760877f,-1.65575026f,
-1.59030191f,-2.09045484f,-2.01113821f,-1.31159853f,-1.3120046f,
-1.45263472f,-1.52268525f,-1.6567962f,-2.06986454f,-1.46546941f,
-1.25549694f,-1.86336982f,-1.64691575f,-1.69584239f,-1.69374889f
		});

	auto logits_length = NDArrayFactory::create<int>('c', { BATCH_LEN }, { MAX_FRAME_LEN });
	auto result_prob = NDArrayFactory::create<float>('c', { BATCH_LEN, NBEST_LEN });
	auto result_seq = NDArrayFactory::create<int>('c', { BATCH_LEN , NBEST_LEN, MAX_FRAME_LEN });
	beamSearch<float, int>(logits, logits_length, result_seq, result_prob, BLANK_INDEX, BEAM_WIDTH, NBEST_LEN);

	result_seq.printIndexedBuffer("result seq");
	result_prob.printIndexedBuffer("result prob");
}



int main(int argc, char** argv)
{

	test123();

	/*int yyy; std::cin >> yyy;
	return -1;*/
	static_assert(std::is_nothrow_move_constructible<SequenceContainer<int>>::value, "");
	static_assert(std::is_nothrow_move_assignable<SequenceContainer<int>>::value, "");
	static_assert(std::is_nothrow_destructible<SequenceContainer<int>>::value, "");

	std::cout << sizeof(SequenceNode<int>) << std::endl;
	const std::vector<float> data = load_file("rnn.csv");

	auto logInputB = NDArrayFactory::create<float>('c', { 1, 100, 80 }, data);

	softmax<float, 3>(logInputB, true);

	const std::string classes = " !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'";

	const auto logInput3 = NDArrayFactory::create<float>('c', { 1, 3, 5 },
														 {
															 -2.578319f, -1.091237f, -1.519336f, -2.115322f, -1.390921f,
															 -1.901657f, -2.46196f, -1.718925f, -0.837558f, -1.874794f,
															 -1.761921f, -1.125581f, -2.378538f, -1.907196f, -1.336974f
														 });
	const auto seq3 = NDArrayFactory::create<int>('c', { 1 }, { 3 });
	auto seq_b = NDArrayFactory::create<int>('c', { 1 }, { 100 });

	auto seq_b2 = NDArrayFactory::create<int>('c', { 1 }, { 0 });
	logInput3.printIndexedBuffer("logInput3");

	testb(beamSearch<float, int>, logInput3, seq3, classes, 3, 2);

	testb(beamSearch<float, int>, logInputB, seq_b, classes, 25, 4);

	testb(beamSearch<float, int>, logInputB, seq_b2, classes, 25, 4);

	auto beam_width = global_beam_width;

	auto batch_len = logInputB.rankOf() > 2 ? logInputB.shapeOf()[0] : 1;
	auto result_prob = NDArrayFactory::create<float>('c', { batch_len, 1 });
	auto result_seq = NDArrayFactory::create<int>('c', { batch_len , 1, 100 });
	beamSearch<float, int>(logInputB, seq_b, result_seq, result_prob, logInputB.shapeOf()[2] - 1, beam_width, 1);
#if  1
	time_it<10, 100>(beamSearch<float, int>, 0, logInputB, seq_b, result_seq, result_prob, logInputB.shapeOf()[2] - 1, beam_width, 1);
#endif

#if  2
	time_it<10, 100>(beamSearch<float, int>, 0, logInputB, seq_b, result_seq, result_prob, logInputB.shapeOf()[2] - 1, beam_width, 1);
#endif


	return -1;
}
