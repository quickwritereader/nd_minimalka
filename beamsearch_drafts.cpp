#include <fstream>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>
#include <numeric>
#include "common.h"
#include "NdArrayMinimal.h"
#include "LoopCoordsHelper.h"
#include "Threads.h"
#include "utils.h"

#include <array>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace sd;

std::vector<float> loadFile(std::string filename)
{
	std::fstream filex(filename);
	std::vector<float> data;
	data.reserve(8 * 1024);
	int colSize = 0;
	bool countedCol = false;
	int countY = 0;
	if (filex.is_open())
	{
		std::string numb;
		while (std::getline(filex, numb, ';'))
		{
			auto pos = numb.find("\n");
			if (pos != std::string::npos)
			{
				//std::cout<<"___"<<numb<< "___";
				numb = numb.erase(pos, 1);
				//std::cout<<"___"<<numb<< "__"<<numb.size()<<std::endl;
				if (!countedCol) colSize = countY;
				countedCol = true;
			}
			++countY;
			//std::cout<<numb<<" ";
			if (numb.size() > 0) data.push_back(std::stof(numb));
		}

		filex.close();
	}
	else std::cout << "Unable to open file";
	std::cout << data.size() << std::endl;
	std::cout << data.data() << std::endl;
	std::cout << data.size() / colSize << " " << colSize << std::endl;
	return data;
}

template <int constRank = 2, bool IsFortran = false>
NDArray Word2LabelSeq(std::string word, std::string classes)
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
	T minVal = std::numeric_limits<T>::min();
	for (size_t i = 0; i < loop_count; i++)
	{
		T* lx = x + offset;
		auto maxVal = lx[0];
		for (size_t j = 0; j <= last_index; j++)
		{
			maxVal = lx[j * el_stride] > maxVal ? lx[j * el_stride] : maxVal;
		}

		T denom_sum = 0;
		for (size_t j = 0; j <= last_index; j++)
		{
			T val = std::exp(lx[j * el_stride] - maxVal);
			denom_sum += val;
			lx[j * el_stride] = val;
		}
		if (!makeLog)
		{
			for (size_t j = 0; j <= last_index; j++)
			{
				lx[j * el_stride] = lx[j * el_stride] / denom_sum;
				if (lx[j * el_stride] < minVal) lx[j * el_stride] = minVal;


				//std::cout<<std::scientific<<lx[j*el_stride]<<", ";
			}
		}
		else
		{
			for (size_t j = 0; j <= last_index; j++)
			{
				lx[j * el_stride] = std::log(lx[j * el_stride]) - std::log(denom_sum);
				//std::cout<<std::scientific<<lx[j*el_stride]<<", ";
			}
		}
		//std::cout<<std::endl;
		offset = sd::inc_coords<constRank, skip, false>(cst, offset);
	}
}


template <typename T>
constexpr T negative_infinity()
{
	return -std::numeric_limits<T>::infinity();
}


template <typename T>
T logSumExp0(T arg1, T arg2)
{
	auto cMax = std::max(arg1, arg2);
	if (cMax == negative_infinity<T>())
	{
		return std::log(std::exp(arg1) + std::exp(arg2));
	}
	return std::log(std::exp(arg1 - cMax) + std::exp(arg2 - cMax)) + cMax;
}

template <typename T>
T log_sum_exp(T arg1, T arg2, T arg3)
{
	auto cMax = std::max(arg1, arg2);
	cMax = std::max(cMax, arg3);
	if (cMax == negative_infinity<T>())
	{
		cMax = 0;
	}
	return std::log(std::exp(arg1 - cMax) + std::exp(arg2 - cMax) + std::exp(arg3 - cMax)) + cMax;
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
	T nonBlank = negative_infinity<T>();
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
struct SequenceNode
{
	//intrusive double links
	SequenceNode<T>* prev = nullptr;
	SequenceNode<T>* next = nullptr;

	//sequence prefix/parent
	SequenceNode<T>* prefix = nullptr;

	T value = DefaultInvalid<T>::value;

	bool leaf = true;

	void markAsFullyExtended()
	{
		leaf = false;
	}

	bool isFullyExtended() const { return leaf; }
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

	SequenceNode<T>* GetLast()
	{
		return current_;
	}

	SequenceNode<T>* extendPath(SequenceNode<T>* prefix, T value)
	{
		auto new_node = new SequenceNode<T>();
		new_node->value = value;
		new_node->prefix = prefix;
		prefix->clearLeaf(); //->leaf = false;

		//add in the holder
		new_node->next = nullptr;
		new_node->prev = current_;

		if (current_) current_->next = new_node;

		current_ = new_node;
		count_++;
		return new_node;
	}

	void remove(SequenceNode<T>* seq)
	{
		if (seq == nullptr || !seq->isLeaf())
		{
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
		int i = 0;
		while (del)
		{
			++i;
			SequenceNode<T>* temp = del->prev;
			delete del;
			del = temp;
		}
		current_ = nullptr;
		assert(count_ == i);
	}

	~SequenceContainer()
	{
		clear();
	}

private:
	SequenceNode<T>* current_ = nullptr;

	int count_ = 0;
};


template <typename T, typename U>
struct BeamEntry
{
	BeamProb<T> prob;
	SequenceNode<U>* sequence;
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

//choose ptr[index*element_stride] 
template <bool StrideAccess, typename Type>
std::enable_if_t<StrideAccess == true, Type&>
element(const Type* ptr, const int index, const int element_stride)
{
	return ptr[index * element_stride];
}

//choose ptr[index] assuming element_stride is 1
template <bool StrideAccess, typename Type>
std::enable_if_t<StrideAccess == false, Type&>
element(const Type* ptr, const int index, const int element_stride)
{
	return ptr[index];
}


void print_seq1(SequenceNode<int>* parent)
{
	auto lf = std::to_string(parent->isFullyExtended());
	auto v = std::string("__leaf");
	std::cout << "{" << (parent->isFullyExtended() ? v : (std::string("__branch ") + lf)) << "__: ";
	for (auto x : SequenceContainer<int>::getSequence(parent))
	{
		std::cout << x << " , ";
	}
	std::cout << "}" << std::endl;
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

template <typename T, int N>
void extend(SequenceContainer<int>& holder, SequenceNode<int>* parent, T(&& seq)[N])
{
	for (auto x : seq)
	{
		parent = holder.extendPath(parent, x);
		print_seq1(parent);
	}
}

constexpr uint32_t bits_required(const uint32_t n)
{
	return n <= 1 ? 0 : 1 + bits_required((n + 1) / 2);
}

struct PairHash
{
	template <class T1, class T2>
	std::size_t operator()(const std::pair<T1*, T2>& p) const
	{
		//return hash_combine(p.first, p.second);
		//first is mostly aligned pointer 
		const auto tmp = (reinterpret_cast<uint64_t>(p.first) >> bits_required(sizeof(T1))) ^ (static_cast<uint64_t>(p.second)
			<< 16);

		return std::hash<uint64_t>()(tmp);
	}
};

template <typename T>
float pr(const int c, const BeamProb<T>& beam_prob, const SequenceNode<int>* seq, const T prob)
{
	return seq->value == c ? beam_prob.blank + prob : beam_prob.total + prob;
}

#pragma region Region_1
std::pair<std::vector<int>, float> beamSearch1(const NDArray& log_input, int blank_index, int beam_width = 25)
{
	auto len_t = log_input.shapeOf()[0];
	auto len_c = log_input.shapeOf()[1];

	auto log_p = log_input.bufferAsT<float>();
	using BeamEntryType = BeamEntry<float, int>;

	//auto batchP = logInput.stridesOf()[0];
	auto inc_p = log_input.stridesOf()[0];
	auto elwise_p = log_input.stridesOf()[1];

	if (beam_width < 1) beam_width = 1;
	SequenceContainer<int> sequence_holder;
	std::vector<BeamEntryType> last_beams;
	BeamEntryType empty;
	empty.prob.blank = 0;
	empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.nonBlank);
	empty.sequence = sequence_holder.GetLast();
	last_beams.push_back(empty);
	for (int t = 0; t < len_t; t++)
	{
		int minCan = std::min<int>(last_beams.size(), beam_width);
		using PairParentChild = std::pair<SequenceNode<int>*, int>;
		std::unordered_map<PairParentChild, int, PairHash> mapIndex;
		std::vector<BeamEntry<float, int>> current_beams;
#if defined(PRINT_DEBUG)
		std::cout << "/////////T: " << t << std::endl;
#endif
		for (int c = 0; c < len_c; c++)
		{
			auto prob = log_p[c]; //element(logP, c, incP)
#if defined(PRINT_DEBUG)
			std::cout << "/////////C: " << c << std::endl;
#endif
			//iterate over beams
			for (int j = 0; j < minCan; j++)
			{
				auto& cur_prob = last_beams[j].prob;
				SequenceNode<int>* seq = last_beams[j].sequence;

				if (c == blank_index)
				{
					auto entryPoint = std::make_pair(seq->prefix, seq->value);
					auto found = mapIndex.find(entryPoint);

					if (found == std::end(mapIndex))
					{
						BeamEntryType entry;
						entry.sequence = seq;
						//logSumExp(entry.prob.blank, currProb.blank + prob, currProb.nonBlank + prob);
						entry.prob.blank = log_sum_exp(cur_prob.blank + prob, cur_prob.nonBlank + prob);
						//logSumExp(entry.prob.blank, entry.prob.nonBlank);
						entry.prob.total = entry.prob.blank;
						current_beams.push_back(entry);
						auto ind = current_beams.size() - 1;
#if  defined(PRINT_DEBUG)
						std::cout << "map#1 " << ind << std::endl;
						print_seq1(seq);
						std::cout << "====m#1" << std::endl;
#endif
						mapIndex[entryPoint] = ind;
					}
					else
					{
#if  defined(PRINT_DEBUG)
						std::cout << "found #1: " << found->second << std::endl;
						print_seq1(seq);
						print_seq1(current_beams[found->second].sequence);
						std::cout << "===f#1" << std::endl;
#endif
						//note: here we took as ref &
						auto& entryProb = current_beams[found->second].prob;
						entryProb.blank = log_sum_exp(entryProb.blank, cur_prob.blank + prob, cur_prob.nonBlank + prob);
						entryProb.total = log_sum_exp(entryProb.blank, entryProb.nonBlank);
					}
				}
				else
				{
					//extend by new character
					auto end_t = seq->value;
					auto entryPoint = std::make_pair(seq, c);
					auto found = mapIndex.find(entryPoint);
					int foundIndex = -1;
					if (found != std::end(mapIndex))
					{
#if  defined(PRINT_DEBUG)
						std::cout << "found #2: " << found->second << std::endl;
						print_seq1(seq, c);
						print_seq1(current_beams[found->second].sequence);
						std::cout << "===f#2" << std::endl;
#endif
						foundIndex = found->second;
					}
					else
					{
						BeamEntryType entry;
						//add new sequence
						SequenceNode<int>* new_sequence;
						new_sequence = sequence_holder.extendPath(seq, c);
						entry.sequence = new_sequence;
						current_beams.push_back(entry);
						foundIndex = current_beams.size() - 1;
#if  defined(PRINT_DEBUG)
						std::cout << "map#2 " << foundIndex << std::endl;
						print_seq1(new_sequence);
						std::cout << "====m#2" << std::endl;
#endif
						mapIndex[entryPoint] = foundIndex;
					}

					BeamProb<float>& new_seq_prob = current_beams[foundIndex].prob;
					if (c != end_t)
					{
						new_seq_prob.nonBlank = log_sum_exp(new_seq_prob.nonBlank, cur_prob.blank + prob,
							cur_prob.nonBlank + prob);
					}
					else
					{
						//c is repeated. skip nonblank
						new_seq_prob.nonBlank = log_sum_exp(new_seq_prob.nonBlank, cur_prob.blank + prob);
					}

					//lm score

					//total
					new_seq_prob.total = log_sum_exp(new_seq_prob.blank, new_seq_prob.nonBlank);

					//if s is repeated at the end we also update the unchanged prefix. merging
					if (c == end_t)
					{
						auto entry_point_prev = std::make_pair(seq->prefix, seq->value);
						found = mapIndex.find(entry_point_prev);
						if (found == std::end(mapIndex))
						{
							BeamEntryType entry;
							entry.sequence = seq;
							//logSumExp(entry.prob.nonBlank, currProb.nonBlank + prob);
							entry.prob.nonBlank = cur_prob.nonBlank + prob;
							//logSumExp(entry.prob.blank, entry.prob.nonBlank);
							entry.prob.total = entry.prob.nonBlank;
							current_beams.push_back(entry);
							auto ind = current_beams.size() - 1;
#if  defined(PRINT_DEBUG)
							std::cout << "map#3 " << ind << std::endl;
							print_seq1(seq);
							std::cout << "====m#3" << std::endl;
#endif
							mapIndex[entry_point_prev] = ind;
						}
						else
						{
#if  defined(PRINT_DEBUG)
							std::cout << "found #3: " << found->second << std::endl;
							print_seq1(seq);
							print_seq1(current_beams[found->second].sequence);
							std::cout << "===f#3" << std::endl;
#endif
							//note: here we took as ref &
							auto& entry_prob = current_beams[found->second].prob;
							entry_prob.nonBlank = log_sum_exp(entry_prob.nonBlank, cur_prob.nonBlank + prob);
							entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.nonBlank);
						}
					}
				}
			} //beams
		} //class
		last_beams = std::move(current_beams);
		log_p += inc_p;

#if 0
		std::nth_element(lastBeams.begin(),
			lastBeams.begin() + beamWidth,
			lastBeams.end(),
			compare_beam_prob<float, int>);
#elif 1
		std::partial_sort(last_beams.begin(),
			last_beams.begin() + beam_width,
			last_beams.end(),
			compare_beam_prob<float, int>);
#else
		std::sort(std::begin(lastBeams), std::end(lastBeams), compare_beam_prob<float, int>);
#endif

		//delete the rest to keep memory efficient
		if (last_beams.size() > beam_width)
		{
			for (int j = beam_width; j < last_beams.size(); j++)
			{
				sequence_holder.remove(last_beams[j].sequence);
			}
			last_beams.erase(std::begin(last_beams) + beam_width, std::end(last_beams));
		}
#if defined(PRINT_DEBUG)
		std::cout << "B////////////////////////////////" << std::endl;
		for (int j = 0; j < std::min((int)last_beams.size(), beam_width); j++) {
			if (!last_beams[j].sequence->isLeaf())
			{
				last_beams[j].sequence->SetLeafVal(j);//set index
			}
			print_seq1(last_beams[j].sequence);

		}
		std::cout << "E////////////////////////////////" << std::endl;
#endif
	}

	std::sort(std::begin(last_beams), std::end(last_beams), compare_beam_prob<float, int>);
	auto top = last_beams[0];
	return { SequenceContainer<int>::getSequence(top.sequence, len_t), top.prob.total };
}

std::pair<std::vector<int>, float> beamSearch2(const NDArray& log_input, const int blank_index, int beam_width = 25)
{
	using PairParentChild = std::pair<SequenceNode<int>*, int>;
	using BeamEntryType = BeamEntry<float, int>;

	const auto len_t = log_input.shapeOf()[0];
	const auto len_c = log_input.shapeOf()[1];

	auto log_p = log_input.bufferAsT<float>();

	//auto batchP = logInput.stridesOf()[0];
	const auto inc_p = log_input.stridesOf()[0];
	auto elwise_p = log_input.stridesOf()[1];

	if (beam_width < 1) beam_width = 1;
	SequenceContainer<int> sequence_container;
	BeamEntryType empty;
	empty.prob.blank = 0;
	empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.nonBlank);
	empty.sequence = sequence_container.GetLast();

	//vectors: we will use it as array, here
	std::vector<BeamEntryType> last_beams, next_beams;
	last_beams.resize(beam_width);
	//as we skip blank indexes reserve count is beam_width * len_c 
	next_beams.resize(beam_width * len_c);
	last_beams[0] = empty;
	auto last_beam_size = 1;

	std::unordered_map<PairParentChild, int, PairHash> map_index;
	map_index.reserve(beam_width * len_c);

	for (int t = 0; t < len_t; t++)
	{
		auto next_beam_size = 0;

		for (auto j = 0; j < last_beam_size; j++)
		{
			SequenceNode<int>* seq = last_beams[j].sequence;
			auto& cur_prob = last_beams[j].prob;
			auto entry_point = std::make_pair(seq->prefix, seq->value);
			//if len(seq) > 0 then
			auto non_blank_prob = seq->value != -1 ? (log_p[seq->value] + cur_prob.nonBlank) : negative_infinity<float>();
			const float blank_prob = log_p[blank_index] + cur_prob.total;

			//check entry
			auto found = map_index.find(entry_point);

			if (found != std::end(map_index))
			{
				//note: here we took as ref &
				auto& entry_prob = next_beams[found->second].prob;
				entry_prob.blank = log_sum_exp(entry_prob.blank, blank_prob);
				entry_prob.nonBlank = log_sum_exp(entry_prob.nonBlank, non_blank_prob);
				entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.nonBlank);
			}
			else
			{
				BeamEntryType entry;
				entry.sequence = seq;
				entry.prob.blank = blank_prob;
				entry.prob.nonBlank = non_blank_prob;
				entry.prob.total = log_sum_exp(blank_prob, non_blank_prob);
				next_beams[next_beam_size] = entry;
				map_index[entry_point] = next_beam_size;

				++next_beam_size;
			}

			for (int c = 0; c < len_c; c++)
			{
				if (c == blank_index) continue;

				const auto prob = log_p[c];
				non_blank_prob = pr(c, cur_prob, seq, prob);

				//extend by new character 
				entry_point = std::make_pair(seq, c);
				found = map_index.find(entry_point);
				if (found != std::end(map_index))
				{
					auto& entry_prob = next_beams[found->second].prob;
					entry_prob.nonBlank = log_sum_exp(entry_prob.nonBlank, non_blank_prob);
					entry_prob.total = log_sum_exp(entry_prob.total, non_blank_prob);
				}
				else
				{
					BeamEntryType entry;
					SequenceNode<int>* new_sequence = sequence_container.extendPath(seq, c);
					entry.prob.nonBlank = non_blank_prob;
					entry.prob.total = non_blank_prob;
					entry.sequence = new_sequence;
					next_beams[next_beam_size] = entry;
					map_index[entry_point] = next_beam_size;
					++next_beam_size;
				}
			} //iteration over classes
		} //iteration over beams


		log_p += inc_p;

		last_beam_size = std::min(next_beam_size, beam_width);

		//sort next beams to get candidates
		std::partial_sort(std::begin(next_beams),
			std::begin(next_beams) + last_beam_size,
			std::begin(next_beams) + next_beam_size, compare_beam_prob<float, int>);


		//copy top beams
		for (int j = 0; j < last_beam_size; j++)
		{
			last_beams[j] = next_beams[j];
		}

		if (t < len_t)
		{
			//reset 
			map_index.clear();
			//delete sequences from the sequence_holder to decrease memory
			for (auto j = beam_width; j < next_beam_size; j++)
			{
				sequence_container.remove(next_beams[j].sequence);
			}
		}

#if defined(PRINT_DEBUG)
		std::cout << "B////////////////////////////////" << std::endl;
		for (size_t j = 0; j < last_beam_size; j++) {
			if (!last_beams[j].sequence->isLeaf())
			{
				last_beams[j].sequence->SetLeafVal(j);//set index
			}
			print_seq1(last_beams[j].sequence);

		}
		std::cout << "E////////////////////////////////" << std::endl;

#endif
	}


	auto top = last_beams[0];
	return { SequenceContainer<int>::getSequence(top.sequence, len_t), top.prob.total };
}
#pragma endregion Region_1


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

template<bool HasElementStride = false, typename Type>
std::pair<std::vector<int>, float> inner_beam_search(Type* log_p, const uint64_t inc_p, const uint64_t len_t, const int len_c, const int blank_index, int beam_width, const uint64_t element_stride = 1L)
{
	using PairParentChild = std::pair<SequenceNode<int>*, int>;
	using BeamEntryType = BeamEntry<Type, int>;
	using BeamEntryTypeEx = BeamEntryEx<Type, int>;
	std::pair<SequenceNode<int>*, int> entry_point;


	if (beam_width < 1) beam_width = 1;
	SequenceContainer<int> sequence_container;
	BeamEntryType empty;
	empty.prob.blank = 0;
	empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.nonBlank);
	empty.sequence = sequence_container.GetLast();

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


	for (int t = 0; t < len_t; t++)
	{
		auto next_beam_size = 0;

		for (auto j = 0; j < last_beam_size; j++)
		{
			SequenceNode<int>* seq = last_beams[j].entry.sequence;
			auto& cur_prob = last_beams[j].entry.prob;
			const OverlapDirection overlap_dir_parent = last_beams[j].overlap_direction & OverlapDirection::PARENT;
			const OverlapDirection overlap_dir_child = last_beams[j].overlap_direction & OverlapDirection::CHILD;
			//if len(seq) > 0 then
			const auto log_p_blank = element<HasElementStride>(log_p, blank_index, element_stride);
			//log_p[seq->value] 
			auto non_blank_prob = seq->value != -1 ? (element<HasElementStride>(log_p, seq->value, element_stride) + cur_prob.nonBlank) : negative_infinity<float>();

			const float blank_prob = log_p_blank + cur_prob.total;

			auto found = std::end(map_index);

			if (overlap_dir_child == OverlapDirection::CHILD)
			{
				//check entry
				entry_point = std::make_pair(seq->prefix, seq->value);
				found = map_index.find(entry_point);
			}

			if (found == std::end(map_index))
			{
				BeamEntryType entry;
				entry.sequence = seq;
				entry.prob.blank = blank_prob;
				entry.prob.nonBlank = non_blank_prob;
				entry.prob.total = log_sum_exp(blank_prob, non_blank_prob);
				next_beams[next_beam_size] = entry;
				//map if its overlapped one. in this case just being child is enough
				if (overlap_dir_child == OverlapDirection::CHILD) map_index[entry_point] = next_beam_size;

				++next_beam_size;
			}
			else
			{
				//note: here we took as ref &
				auto& entry_prob = next_beams[found->second].prob;
				entry_prob.blank = log_sum_exp(entry_prob.blank, blank_prob);
				entry_prob.nonBlank = log_sum_exp(entry_prob.nonBlank, non_blank_prob);
				entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.nonBlank);
			}

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
					SequenceNode<int>* new_sequence = sequence_container.extendPath(seq, c);
					entry.prob.nonBlank = non_blank_prob;
					entry.prob.total = non_blank_prob;
					entry.sequence = new_sequence;
					next_beams[next_beam_size] = entry;

					//we will map only ones that can overlap
					if (overlap_dir_parent == OverlapDirection::PARENT && overlap_classes[c]) map_index[entry_point] =
						next_beam_size;
					++next_beam_size;
				}
				else
				{
					auto& entry_prob = next_beams[found->second].prob;
					entry_prob.nonBlank = log_sum_exp(entry_prob.nonBlank, non_blank_prob);
					entry_prob.total = log_sum_exp(entry_prob.total, non_blank_prob);
				}
			} //iteration over classes
		} //iteration over  beams

		log_p += inc_p;

		last_beam_size = std::min(next_beam_size, beam_width);

		//sort next beams to get candidates
		std::partial_sort(std::begin(next_beams),
			std::begin(next_beams) + last_beam_size,
			std::begin(next_beams) + next_beam_size, compare_beam_prob<float, int>);

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
			for (int j = 0; j < len_c; j++)
			{
				overlap_classes[j] = false;
			}

			//check overlap_direction
			for (int j = 0; j < last_beam_size; j++)
			{
				if (!last_beams[j].entry.sequence->isLeaf())
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


	auto top = next_beams[0];
	return { SequenceContainer<int>::getSequence(top.sequence, len_t), -top.prob.total };
}

/**
 * inputs	3-D float Tensor, size [max_time, batch_size, num_classes]. The logits.
sequence_length	1-D int32 vector containing sequence lengths, having size [batch_size].
beam_width	An int scalar >= 0 (beam search beam width).
top_paths	An int scalar >= 0, <= beam_width (controls output size).
 */

template<typename Type, typename IndexType = int>
std::vector<std::pair<std::vector<int>, Type>>
beamSearch(const NDArray& logits, const NDArray& sequence_length, int blank_index, int beam_width = 25)
{


	const auto logits_shapes = logits.shapeOf();
	const auto logit_strides = logits.stridesOf();
	const auto rank = logits.rankOf();

	IndexType* len_t_ptr = nullptr;
	int element_stride_t = 1;;

	//checks before
	if (rank < 2) return {};
	auto batch_len = rank > 2 ? logits_shapes[0] : 1;
	auto max_len_t = logits_shapes[rank - 2];
	auto len_c = logits_shapes[rank - 1];

	if (len_c < 1 || max_len_t < 1) return {};
	if (blank_index > len_c || blank_index < 0) blank_index = static_cast<int>(len_c) - 1;
	if (sequence_length.rankOf() == 1 && sequence_length.shapeOf()[0] == batch_len)
	{
		len_t_ptr = sequence_length.bufferAsT<IndexType>();
		element_stride_t = sequence_length.stridesOf()[0];
	}

	//strides
	auto batch_stride = rank > 2 ? logit_strides[0] : 0;
	auto inc_p = logit_strides[rank - 2];
	auto element_stride = logits.stridesOf()[rank - 1];

	auto logits_ptr = logits.bufferAsT<Type>();


	std::vector<std::pair<std::vector<int>, Type>> results;
	results.resize(batch_len);
	auto result_ptr = results.data();

	auto func = [result_ptr, batch_len, max_len_t, len_c, batch_stride, inc_p, element_stride, element_stride_t, logits_ptr, len_t_ptr, blank_index, beam_width]
	(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void
	{

		auto ptr = logits_ptr + start * batch_stride;

		if (element_stride == 1)
		{
			//choose ews one
			for (auto b = start; b < stop; b += increment)
			{
				auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
				len_t = len_t > max_len_t ? max_len_t : len_t;
				if (len_t > 0)
				{
					result_ptr[b] = inner_beam_search<false, Type>(ptr, inc_p, len_t, len_c, blank_index, beam_width);
				}
				ptr += batch_stride;
			}
		}
		else
		{
			// element with stride case 
			for (auto b = start; b < stop; b += increment)
			{
				auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
				len_t = len_t > max_len_t ? max_len_t : len_t;
				if (len_t > 0)
				{
					result_ptr[b] = inner_beam_search<true, Type>(ptr, inc_p, len_t, len_c, blank_index, beam_width, element_stride);
				}
				ptr += batch_stride;
			}
		}
	};
	samediff::Threads::parallel_for(func, 0, batch_len, 1);
	return results;
}

template <typename Op>
void test1(Op op, const NDArray& logInput, const std::string& classes, int w = 3)
{
	auto blankInd = logInput.shapeOf()[1] - 1;
	std::cout << "OP: blankInd " << blankInd << ", " << w << std::endl;
	auto result = op(logInput, blankInd, w);

	std::cout << "log prob " << result.second << std::endl;
	auto lblIndex = result.first;
	std::string s;
	for (auto x : lblIndex)
	{
		s += classes[x];
		std::cout << x << ",";
	}
	std::cout << "\n" << s << std::endl;
}

template <typename Op >
void test_new(Op op, const NDArray& logInput, const NDArray& seq_len, const std::string& classes, int w = 3)
{
	int blankInd = logInput.shapeOf()[logInput.rankOf() - 1] - 1;
	std::cout << "OP: blankInd " << blankInd << ", " << w << std::endl;
	std::vector<std::pair<std::vector<int>, float>> results = op(logInput, seq_len, blankInd, w);
	for (auto& result : results) {
		std::cout << "log prob " << result.second << std::endl;
		auto lblIndex = result.first;
		std::string s;
		for (auto x : lblIndex)
		{
			s += classes[x];
			std::cout << x << ",";
		}
		std::cout << "\n" << s << std::endl;
	}
}


constexpr int global_beam_width = 25;

extern "C" float test_new()
{
	static NDArray logInput;
	static int t = 0;
	const std::string classes = " !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'";
	if (t == 0)
	{
		const std::vector<float> data = loadFile("rnn.csv");
		logInput = NDArrayFactory::create<float>('c', { 100, 80 }, data);
		softmax<float, 2>(logInput, true);
		t = 1;
	}


	const auto result = beamSearch2(logInput, logInput.shapeOf()[1] - 1, global_beam_width);
	return result.second;
}


extern "C" float test_new_improved()
{
	static NDArray logInput, seq_length;
	static int t = 0;
	const std::string classes = " !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'";
	if (t == 0)
	{
		const std::vector<float> data = loadFile("rnn.csv");
		logInput = NDArrayFactory::create<float>('c', { 1, 100, 80 }, data);
		seq_length = NDArrayFactory::create<int>('c', { 1 }, { 100 });
		softmax<float, 3>(logInput, true);
		t = 1;
	}


	auto results = beamSearch<float, int>(logInput, seq_length, logInput.shapeOf()[1] - 1, global_beam_width);
	return results[0].second;
}


extern "C" float test_old()
{
	static NDArray logInput;
	static int t = 0;
	const std::string classes = " !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'";
	if (t == 0)
	{
		const std::vector<float> data = loadFile("rnn.csv");
		logInput = NDArrayFactory::create<float>('c', { 100, 80 }, data);
		softmax<float, 2>(logInput, true);
		t = 1;
	}


	const auto result = beamSearch1(logInput, logInput.shapeOf()[1] - 1, global_beam_width);
	return result.second;
}

int main(int argc, char** argv)
{
	//{
	//	SequenceContainer<int> holder;
	//	SequenceNode<int>* parent = holder.getLast();
	//	//add () 
	//	int sequence[] = { 3, 4,5,6,7,12,14 };

	//	for (auto x : sequence) {
	//		parent = holder.add(parent, x);
	//		extend(holder, parent, { x*10+1,x * 10 + 2,x * 10 + 3,x * 10 + 4,x * 10 + 5,x * 10 + 6 });
	//		print_seq(holder, parent);
	//	}
	//	print_seq(holder, parent);

	//} 
	static_assert(std::is_nothrow_move_constructible<SequenceContainer<int>>::value, "");
	static_assert(std::is_nothrow_move_assignable<SequenceContainer<int>>::value, "");
	static_assert(std::is_nothrow_destructible<SequenceContainer<int>>::value, "");

	std::cout << sizeof(SequenceNode<int>) << std::endl;
	const std::vector<float> data = loadFile("rnn.csv");

	auto logInput = NDArrayFactory::create<float>('c', { 100, 80 }, data);

	softmax<float, 2>(logInput, true);


	auto logInputB = NDArrayFactory::create<float>('c', { 1, 100, 80 }, data);

	softmax<float, 3>(logInputB, true);


	const std::string classes = " !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'";


	auto logInput2 = NDArrayFactory::create<float>('c', { 3, 5 });
	fill_nd<float>(logInput2);
	softmax<float, 2>(logInput2, true);


	const auto logInput3 = NDArrayFactory::create<float>('c', { 1, 3, 5 },
														 {
															 -2.578319f, -1.091237f, -1.519336f, -2.115322f, -1.390921f,
															 -1.901657f, -2.46196f, -1.718925f, -0.837558f, -1.874794f,
															 -1.761921f, -1.125581f, -2.378538f, -1.907196f, -1.336974f
														 });
	auto seq3 = NDArrayFactory::create<int>('c', { 1 }, { 3 });
	auto seqB = NDArrayFactory::create<int>('c', { 1 }, { 100 });

	logInput3.printIndexedBuffer("logInput3");

	test_new(beamSearch<float, int>, logInput3, seq3, classes, 3);

	test_new(beamSearch<float, int>, logInputB, seqB, classes, 25);
	auto beam_width = global_beam_width;
#if  0
	time_it<10, 100>(beamSearch1, 0, logInput, logInput.shapeOf()[1] - 1, beam_width);
#endif

#if  1
	time_it<10, 100>(beamSearch<float, int>, 0, logInputB, seqB, logInput.shapeOf()[2] - 1, beam_width);
#endif

#if  2
	time_it<10, 100>(beamSearch<float, int>, 0, logInputB, seqB, logInput.shapeOf()[2] - 1, beam_width);
#endif

#if  0
	time_it<10, 100>(beamSearch2, 0, logInput, logInput.shapeOf()[1] - 1, beam_width);
#endif

#if  0
	time_it<10, 100>(test_new, 0);
#endif
	//int y;
	//std::cin >> y;
	return -1;
}


#if 0

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
struct SequenceNode
{
	//intrusive double links
	SequenceNode<T>* prev = nullptr;
	SequenceNode<T>* next = nullptr;

	//sequence prefix/parent
	SequenceNode<T>* prefix = nullptr;

	T value = DefaultInvalid<T>::value;

	int state = 0;

	void markAsExtended()
	{
		state |= 1;
	}

	void markAsShared()
	{
		state |= 2;
	}
	bool isShared() const
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

	SequenceNode<T>* getLast()
	{
		return current_;
	}

	SequenceNode<T>* add(SequenceNode<T>* prefix, T value)
	{
		auto new_node = new SequenceNode<T>();
		new_node->value = value;
		new_node->prefix = prefix;


		//add in the holder
		new_node->next = nullptr;
		new_node->prev = current_;

		if (current_) current_->next = new_node;

		current_ = new_node;
		count_++;
		return new_node;
	}

	void remove(SequenceNode<T>* seq)
	{
		if (seq == nullptr || seq->isFullyExtended() || seq->isShared())
		{
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

		delete seq;
		count_--;
	}

	static std::vector<T> getNodeSequence(SequenceNode<T>* seq, size_t reserve_size = 1024)
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


void print_seq1(SequenceNode<int>* parent)
{
	//const auto lf = std::to_string(parent->isFullyExtended());
	//const auto v = std::string("__leaf");
	//if (parent->isFullyExtended())
	//	std::cout << "{" << v << "__: ";
	//else
	//	std::cout << "{__branch " << lf << "__: ";
	std::cout << "(";
	for (auto x : SequenceContainer<int>::getNodeSequence(parent))
	{
		std::cout << x << ", ";
	}
	std::cout << ")" << std::endl;
}

void print_seq1(SequenceNode<int>* parent, int value)
{
	std::cout << "{";
	for (auto x : SequenceContainer<int>::getNodeSequence(parent))
	{
		std::cout << x << " , ";
	}
	std::cout << value << " , ";
	std::cout << "}" << std::endl;
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
						//try to find sequence in time frame t beams
						for (int tb = 0; tb < last_beam_size; tb++)
						{
							auto ss = last_beams[tb].entry.sequence;
							if (ss->prefix == seq && ss->value == c)
							{
								new_sequence = ss; //already extended one for this time frame;
								ss->markAsShared();
								break;
							}
						}

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
			print_seq1(next_beams[j].sequence);
			std::cout << std::setprecision(17) << next_beams[j].prob.total << std::endl;
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


#endif