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
class SequenceContainer;


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
		//we will have just two copies in bad case. so just or
		state = state | 2;
	}

	void decreaseRef()
	{
		//we will have just two cases in bad case, so just remove that
		state = state & (-2);
	}

	bool safeToRemove()
	{

		if (state & 1) return false;

		decreaseRef();
		//we do not want to remove parent nodes in our case. otherwise just returning state<=1 is ok
		return state == 0;
	}

	bool isFullyExtended() const { return state & 1; }
};

/***
 * Sequence container.
 *
 * NOTE: it is not thread-safe
 *
 * Extend path - O(1)
 * Remove path - O(1)
 * Generating Sequence with backtracking prefix:  O(n)
 *
 * Note: Sequence container is implemented primitively and only usable within this task.
 * As it does not behave as a fully capable tree. some cases should be handled manually
 *
 * Here is special cases that should be handled manually to exploit tree/graph behaviour:
 *
 *   Extending new path value:
 *
 *		To extend the path one need to give path and value and in return get new_path:
 *			new_path = container.extendPath ( path, new_value );
 *
 *		Also note that:
 *		SequenceContainer has already default empty path as a beginning point for paths.
 *		So as an initial node one should use it.
 *		   initial_path = container.getEmptyPath();
 *
 *   Adding new path that could be already in container:
 *
 *      Assume we have two paths that can overlap in next step
 *      1st path: node#0() -> node#1(1)                   => generated sequence {},{1}
 *      2nd path: node#0() -> node#1(1) -> node#2(2)      => generated sequence {},{1}, {2}
 *
 *      While extending the first path with value (2). it will be:
 *
 *      node#0() -> node#0(1) -> node#( either new or old)(2)       => generated sequence {},{1}, {2}
 *
 *      For some tasks its not desired to have additional node that will generate the same sequence.
 *      For example:
 *        Assume you wanted to use it as sequence entry in map with just (entry->prefix, entry->value).
 *        so in that case having different paths is not correct and will not be unique in map.
 *
 *      there is not direct way to handle that in our container other than searching.
 *      So one should look for the node with prefix node#1(1) and value(2) and return that node instead of adding new one

 *      Fortunately, for our beam search case:
 *
 *      we need only look for such overlapped cases within the candidates list.
 *      which makes it easy to determine them beforehand while finding and marking overlapped cases. instead of looking for it in SequenceContainer
 *
 *   Removing the same nodes multiple times:
 *		It is fast to remove nodes. As nodes can be stored externally One should follow this rule:
 *
 *		One should not remove the same node twice as it will lead to double free. as Nodes are pointers the same applies to removing a copy
 *
 *		There could be cases where you would like to store copy of nodes. in that cases you can use below method to be able to safely remove:
 *		   node should have mutable method named safeToRemove().
 *		   Basic implementation will be decreasing reference/copy counts and returning true if it is safe to delete
 *
 *
 */
template <typename T>
class SequenceContainer
{
public:
	SequenceContainer() : count_(1)
	{
		empty_path = new SequenceNode<T>();
		current_ = empty_path;
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

		if (!seq->safeToRemove()) return;

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

	SequenceNode<T>* empty_path = nullptr;

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
	//keep indices for lookUp
	int index_as_child = -1;
	int index_as_parent = -1;
	int children_count = 0;
};


template <typename T, typename U>
struct LookUpEntry
{
	U last_c;  //this is is the same as node->value. just we added for the speed
	SequenceNode<U>* node = nullptr;
	int next_beam_index = -1; //index inside next_beam array
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
void inner_beam_search_old(const Type* log_p, const uint64_t inc_p, IndexType* result_sequence, const uint64_t inc_res_seq, const uint64_t max_len_t, Type* result_prob, IndexType* result_seq_length, uint64_t len_t, const uint64_t len_c, const int blank_index, int beam_width, int nbest_len, const uint64_t element_stride = 1L)
{

	using BeamEntryType = BeamEntry<Type, IndexType>;
	using BeamEntryTypeEx = BeamEntryEx<Type, IndexType>;

	if (beam_width < 1) beam_width = 1;
	if (nbest_len > beam_width) nbest_len = beam_width;
	//if len_t is greater than max_len_t truncate it
	len_t = len_t > max_len_t ? max_len_t : len_t;

	SequenceContainer<IndexType> sequence_container;
	BeamEntryType empty;
	empty.prob.blank = 0;
	empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.non_blank);
	empty.sequence = sequence_container.getEmptyPath();

	//vectors: we will use it as array, here
	std::vector<BeamEntryTypeEx> last_beams;
	std::vector<BeamEntryType> next_beams;
	last_beams.resize(beam_width);
	//as we skip blank indexes the count is beam_width * len_c 
	next_beams.resize(beam_width * len_c);
	last_beams[0] = { empty, -1, -1, 0 };
	auto last_beam_size = 1;

	// lookupContainer.
	// considering in each step there will be overlapped cases
	// the size of overlapped cases inside last_beam[0:beam_width]:
	//    assuming we have beam_width size in each step after sort prunning
	//    there is at least one item who will not have any parent
	//    so for the rest (beam_width-1) it will be  has_parent_in_container() ? 1 : 0
	//    so maximum we can have beam_width-1 overlapping parent child pairs

	std::vector<LookUpEntry<Type, IndexType>> lookUp;
	lookUp.resize(beam_width - 1);

	//additional performance flag improvement
	std::vector<IndexType> child_class_hits;
	child_class_hits.resize(len_c);
	for (int c = 0; c < len_c; c++)
	{
		child_class_hits[c] = 0;
	}

	for (uint64_t t = 0; t < len_t; t++)
	{
		auto next_beam_size = 0;
		//std::cout << "---" << t << "---" << std::endl;
		for (auto j = 0; j < last_beam_size; j++)
		{
			SequenceNode<IndexType>* seq = last_beams[j].entry.sequence;
			auto& cur_prob = last_beams[j].entry.prob;
			//if len(seq) > 0 then
			const auto log_p_blank = element<HasElementStride>(log_p, blank_index, element_stride);

			//log_p[seq->value] 
			auto non_blank_prob = seq->value != -1 ? (element<HasElementStride>(log_p, seq->value, element_stride) + cur_prob.non_blank) : negative_infinity<Type>();

			const Type blank_prob = log_p_blank + cur_prob.total;


			auto look_up_beam_index = -1;

			if (last_beams[j].index_as_child != -1)
			{
				//check entry
				look_up_beam_index = lookUp[last_beams[j].index_as_child].next_beam_index;
			}

			if (look_up_beam_index == -1)
			{
				BeamEntryType entry;
				entry.sequence = seq;
				entry.prob.blank = blank_prob;
				entry.prob.non_blank = non_blank_prob;
				entry.prob.total = log_sum_exp(blank_prob, non_blank_prob);
				next_beams[next_beam_size] = entry;
				//map if its overlapped one. in this case just being child is enough
				if (last_beams[j].index_as_child != -1)
				{
					lookUp[last_beams[j].index_as_child].next_beam_index = next_beam_size;
				}
				++next_beam_size;
			}
			else
			{
				//note: here we took as ref &
				auto& entry_prob = next_beams[look_up_beam_index].prob;
				entry_prob.blank = log_sum_exp(entry_prob.blank, blank_prob);
				entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
				entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.non_blank);
			}
			//check to see if it is overlapped parent and possible hits
			auto possible_hit_count = last_beams[j].children_count;

			for (int c = 0; c < len_c; c++)
			{
				if (c == blank_index) continue;

				const auto prob = element<HasElementStride>(log_p, c, element_stride);//log_p[c];

				non_blank_prob = pr(c, cur_prob, seq, prob);


				//extend by new character 
				auto look_up_beam_index_ex = -1;
				SequenceNode<IndexType>* shared_extended_sequence = nullptr;
				int found_index = -1;
				if (possible_hit_count > 0 && child_class_hits[c] > 0)
				{
					//get index within array
					for (int l = last_beams[j].index_as_parent; l < last_beams[j].index_as_parent + last_beams[j].children_count; ++l)
					{
						if (lookUp[l].last_c == c)
						{
							look_up_beam_index_ex = lookUp[l].next_beam_index;
							shared_extended_sequence = lookUp[l].node;
							found_index = l;

							//child_class_hits decrease
							child_class_hits[c] = child_class_hits[c] - 1;
							break;
						}
					}
				}

				if (look_up_beam_index_ex == -1)
				{
					BeamEntryType entry;
					SequenceNode<IndexType>* extended_sequence;
					if (shared_extended_sequence)
					{
						extended_sequence = shared_extended_sequence;
						shared_extended_sequence->increaseRef();
						//assing next_beam_index for lookup
						lookUp[found_index].next_beam_index = next_beam_size;
						//decrease possible_hit_count count
						possible_hit_count -= 1;
					}
					else {
						extended_sequence = sequence_container.extendPath(seq, c);
					}
					entry.prob.non_blank = non_blank_prob;
					entry.prob.total = non_blank_prob;
					entry.sequence = extended_sequence;
					next_beams[next_beam_size] = entry;

					++next_beam_size;
				}
				else
				{
					auto& entry_prob = next_beams[look_up_beam_index_ex].prob;
					entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
					entry_prob.total = log_sum_exp(entry_prob.total, non_blank_prob);
				}
			} //iteration over classes

			//mark it as extended
			seq->markAsFullyExtended();

		} //iteration over  beams

		log_p += inc_p;

#if 0
		std::cout << "---" << t << "---" << std::endl;
		//print
		for (int j = 0; j < next_beam_size; j++)
		{
			//std::cout << " in container " << (long long)next_beams[j].sequence << std::endl;
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
#if 0
		std::cout << "--candidates-" << t << "---" << std::endl;
		//print
		for (int j = 0; j < last_beam_size; j++)
		{
			//std::cout << " in container " << (long long)next_beams[j].sequence << std::endl;
			print_seq1(next_beams[j].sequence);
			std::cout << std::setprecision(17) << next_beams[j].prob.total << std::endl;
		}
		std::cout << "\n\n\n" << std::endl;
#endif		

		if (t < len_t)
		{
			//copy top beams
			for (int j = 0; j < last_beam_size; j++)
			{
				last_beams[j].entry = next_beams[j];
				last_beams[j].index_as_child = -1;
				last_beams[j].index_as_parent = -1;
				last_beams[j].children_count = 0;
			}

			int look_up_index = 0;

			//delete sequences from the sequence_holder to decrease memory
			for (auto j = beam_width; j < next_beam_size; j++)
			{
				sequence_container.remove(next_beams[j].sequence);
			}

			//check overlap_direction
			for (auto j = 0; j < last_beam_size; j++)
			{
				//if it is not parent node then there is not any need to check its possible children
				if (last_beams[j].entry.sequence->isFullyExtended())
				{

					int children_count = 0;

					for (int k = 0; k < last_beam_size; k++)
					{
						if (last_beams[k].entry.sequence->prefix == last_beams[j].entry.sequence)
						{
							if (children_count == 0) {
								last_beams[j].index_as_parent = look_up_index;
							}
							auto class_index = last_beams[k].entry.sequence->value;
							lookUp[look_up_index].node = last_beams[k].entry.sequence;
							lookUp[look_up_index].last_c = class_index;
							lookUp[look_up_index].next_beam_index = -1;
							last_beams[k].index_as_child = look_up_index++;
							child_class_hits[class_index] = child_class_hits[class_index] + 1;
							children_count++;
						}
					}

					if (children_count > 0)
					{
						last_beams[j].children_count = children_count;
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
			auto result_vector = SequenceContainer<IndexType>::getSequence(top.sequence, len_t);
			const auto seq_size = result_vector.size();

			result_prob[j] = top.prob.total;
			result_seq_length[j] = seq_size;
			//copy sequence
			for (auto s = 0; s < seq_size; s++)
			{
				result_sequence[s] = result_vector[s];
			}

			result_sequence += inc_res_seq;

		}
	}
	else
	{
		for (int j = 0; j < nbest_len; j++)
		{
			result_prob[j] = negative_infinity<Type>();
			result_seq_length[j] = 0;;
		}
	}
	return;
}



template<bool HasElementStride = false, typename Type, typename IndexType>
void inner_beam_search_2(const Type* log_p, const uint64_t inc_p, IndexType* result_sequence, const uint64_t inc_res_seq, const uint64_t max_len_t, Type* result_prob, IndexType* result_seq_length, uint64_t len_t, const uint64_t len_c, const int blank_index, int beam_width, int nbest_len, const uint64_t element_stride = 1L)
{

	using BeamEntryType = BeamEntry<Type, IndexType>;
	using BeamEntryTypeEx = BeamEntryEx<Type, IndexType>;

	if (beam_width < 1) beam_width = 1;
	if (nbest_len > beam_width) nbest_len = beam_width;
	//if len_t is greater than max_len_t truncate it
	len_t = len_t > max_len_t ? max_len_t : len_t;

	SequenceContainer<IndexType> sequence_container;
	BeamEntryType empty;
	empty.prob.blank = 0;
	empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.non_blank);
	empty.sequence = sequence_container.getEmptyPath();

	//vectors: we will use it as array, here
	std::vector<BeamEntryTypeEx> last_beams;
	std::vector<BeamEntryType> next_beams;
	last_beams.resize(beam_width);
	//as we skip blank indexes the count is beam_width * len_c 
	next_beams.resize(beam_width * len_c);
	last_beams[0] = { empty, -1, -1, 0 };
	auto last_beam_size = 1;

	// lookupContainer.
	// considering in each step there will be overlapped cases
	// the size of overlapped cases inside last_beam[0:beam_width]:
	//    assuming we have beam_width size in each step after sort prunning
	//    there is at least one item who will not have any parent
	//    so for the rest (beam_width-1) it will be  has_parent_in_container() ? 1 : 0
	//    so maximum we can have beam_width-1 overlapping parent child pairs

	std::vector<LookUpEntry<Type, IndexType>> lookUp;
	lookUp.resize(beam_width - 1);

	//additional storage to sort overlapped case by classes
	std::vector<int> child_class_sorter_help;
	child_class_sorter_help.resize(len_c);

	for (int c = 0; c < len_c; c++)
	{
		child_class_sorter_help[c] = -1;
	}

	for (uint64_t t = 0; t < len_t; t++)
	{
		auto next_beam_size = 0;
		//std::cout << "---" << t << "---" << std::endl;
		for (auto j = 0; j < last_beam_size; j++)
		{
			SequenceNode<IndexType>* seq = last_beams[j].entry.sequence;
			auto& cur_prob = last_beams[j].entry.prob;
			//if len(seq) > 0 then
			const auto log_p_blank = element<HasElementStride>(log_p, blank_index, element_stride);

			//log_p[seq->value] 
			auto non_blank_prob = seq->value != -1 ? (element<HasElementStride>(log_p, seq->value, element_stride) + cur_prob.non_blank) : negative_infinity<Type>();

			const Type blank_prob = log_p_blank + cur_prob.total;


			auto look_up_beam_index = -1;

			if (last_beams[j].index_as_child != -1)
			{
				//check entry
				look_up_beam_index = lookUp[last_beams[j].index_as_child].next_beam_index;
			}

			if (look_up_beam_index == -1)
			{
				BeamEntryType entry;
				entry.sequence = seq;
				entry.prob.blank = blank_prob;
				entry.prob.non_blank = non_blank_prob;
				entry.prob.total = log_sum_exp(blank_prob, non_blank_prob);
				next_beams[next_beam_size] = entry;
				//map if its overlapped one. in this case just being child is enough
				if (last_beams[j].index_as_child != -1)
				{
					lookUp[last_beams[j].index_as_child].next_beam_index = next_beam_size;
				}
				++next_beam_size;
			}
			else
			{
				//note: here we took as ref &
				auto& entry_prob = next_beams[look_up_beam_index].prob;
				entry_prob.blank = log_sum_exp(entry_prob.blank, blank_prob);
				entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
				entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.non_blank);
			}
			//check to see if it is overlapped parent and possible hits
			auto start_index = last_beams[j].index_as_parent;
			auto end_index = last_beams[j].index_as_parent + last_beams[j].children_count;

			for (int c = 0; c < len_c; c++)
			{
				if (c == blank_index) continue;

				const auto prob = element<HasElementStride>(log_p, c, element_stride);//log_p[c];

				non_blank_prob = pr(c, cur_prob, seq, prob);


				//extend by new character 
				auto look_up_beam_index_ex = -1;
				SequenceNode<IndexType>* shared_extended_sequence = nullptr;
				int found_index = -1;
				if (start_index < end_index)
				{
					//get index within array if its that class index
					if (lookUp[start_index].last_c == c)
					{
						look_up_beam_index_ex = lookUp[start_index].next_beam_index;
						shared_extended_sequence = lookUp[start_index].node;
						found_index = start_index;
						++start_index;
					}
				}

				if (look_up_beam_index_ex == -1)
				{
					BeamEntryType entry;
					SequenceNode<IndexType>* extended_sequence;
					if (shared_extended_sequence)
					{
						extended_sequence = shared_extended_sequence;
						shared_extended_sequence->increaseRef();
						//assing next_beam_index for lookup
						lookUp[found_index].next_beam_index = next_beam_size;
					}
					else {
						extended_sequence = sequence_container.extendPath(seq, c);
					}
					entry.prob.non_blank = non_blank_prob;
					entry.prob.total = non_blank_prob;
					entry.sequence = extended_sequence;
					next_beams[next_beam_size] = entry;

					++next_beam_size;
				}
				else
				{
					auto& entry_prob = next_beams[look_up_beam_index_ex].prob;
					entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
					entry_prob.total = log_sum_exp(entry_prob.total, non_blank_prob);
				}
			} //iteration over classes

			//mark it as extended
			seq->markAsFullyExtended();

		} //iteration over  beams

		log_p += inc_p;

#if 0
		std::cout << "---" << t << "---" << std::endl;
		//print
		for (int j = 0; j < next_beam_size; j++)
		{
			//std::cout << " in container " << (long long)next_beams[j].sequence << std::endl;
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
#if 0
		std::cout << "--candidates-" << t << "---" << std::endl;
		//print
		for (int j = 0; j < last_beam_size; j++)
		{
			//std::cout << " in container " << (long long)next_beams[j].sequence << std::endl;
			print_seq1(next_beams[j].sequence);
			std::cout << std::setprecision(17) << next_beams[j].prob.total << std::endl;
		}
		std::cout << "\n\n\n" << std::endl;
#endif		

		if (t < len_t)
		{
			//copy top beams
			for (int j = 0; j < last_beam_size; j++)
			{
				last_beams[j].entry = next_beams[j];
				last_beams[j].index_as_child = -1;
				last_beams[j].index_as_parent = -1;
				last_beams[j].children_count = 0;
			}


			//delete sequences from the sequence_holder to decrease memory
			for (auto j = beam_width; j < next_beam_size; j++)
			{
				sequence_container.remove(next_beams[j].sequence);
			}


			//check overlap_direction and create lookUp with sorted classes as well

			int look_up_index = 0;
			//int lenc_4x = len_c / 4;
			//int lenc_tail = len_c & (-4);
			for (auto j = 0; j < last_beam_size; j++)
			{
				//if it is not parent node then there is not any need to check its possible children
				if (last_beams[j].entry.sequence->isFullyExtended())
				{

					int children_count = 0;
					for (int k = 0; k < last_beam_size; k++)
					{
						auto current = last_beams[k].entry.sequence;
						if (current->prefix == last_beams[j].entry.sequence)
						{

							auto class_index = current->value;
							child_class_sorter_help[class_index] = k;
							children_count++;
						}
					}

					if (children_count > 0)
					{
						//arrange and set. this way we will get sorted children by class
						//and reference to them inside last_beams

						last_beams[j].index_as_parent = look_up_index;
						last_beams[j].children_count = children_count;

						for (int c = 0; c < len_c; c++)
						{
							int k = child_class_sorter_help[c];
							if (k != -1)
							{
								last_beams[k].index_as_child = look_up_index;
								auto seq = last_beams[k].entry.sequence;
								lookUp[look_up_index].last_c = seq->value;
								lookUp[look_up_index].node = seq;
								lookUp[look_up_index].next_beam_index = -1;


								//reset to make child_class_sorter_help usable again for that class index
								child_class_sorter_help[c] = -1;

								//next one
								++look_up_index;
							}
						}
					}//add sorted lookUps

				}
			} //overlap_direction identified to speed up lookUp
		}
	}

	//store nbest results
	if (nbest_len <= last_beam_size) {
		for (int j = 0; j < nbest_len; j++)
		{
			auto top = next_beams[j];
			auto result_vector = SequenceContainer<IndexType>::getSequence(top.sequence, len_t);
			const auto seq_size = result_vector.size();

			result_prob[j] = top.prob.total;
			result_seq_length[j] = seq_size;
			//copy sequence
			for (auto s = 0; s < seq_size; s++)
			{
				result_sequence[s] = result_vector[s];
			}

			result_sequence += inc_res_seq;

		}
	}
	else
	{
		for (int j = 0; j < nbest_len; j++)
		{
			result_prob[j] = negative_infinity<Type>();
			result_seq_length[j] = 0;;
		}
	}
	return;
}


template<bool HasElementStride = false, typename Type, typename IndexType>
void inner_beam_search(const Type* log_p, const uint64_t inc_p, IndexType* result_sequence, const uint64_t inc_res_seq, const uint64_t max_len_t, Type* result_prob, IndexType* result_seq_length, uint64_t len_t, const uint64_t len_c, const int blank_index, int beam_width, int nbest_len, const uint64_t element_stride = 1L)
{

	using BeamEntryType = BeamEntry<Type, IndexType>;
	using BeamEntryTypeEx = BeamEntryEx<Type, IndexType>;

	if (beam_width < 1) beam_width = 1;
	if (nbest_len > beam_width) nbest_len = beam_width;
	//if len_t is greater than max_len_t truncate it
	len_t = len_t > max_len_t ? max_len_t : len_t;

	SequenceContainer<IndexType> sequence_container;
	BeamEntryType empty;
	empty.prob.blank = 0;
	empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.non_blank);
	empty.sequence = sequence_container.getEmptyPath();

	//vectors: we will use it as array, here
	std::vector<BeamEntryTypeEx> last_beams;
	std::vector<BeamEntryType> next_beams;
	last_beams.resize(beam_width);
	//as we skip blank indexes the count is beam_width * len_c 
	next_beams.resize(beam_width * len_c);
	last_beams[0] = { empty, -1, -1, 0 };
	auto last_beam_size = 1;

	// lookupContainer:
	// it will keep sorted entries. so we will just move and compare the entry
	// in each step there will be overlapped cases
	// the size of overlapped cases in last_beam[0:beam_width]:
	//    as we have beam_width size in each step after sort and pruning
	//    there is at least one item who will not have any parent
	//    and for the rest (beam_width-1) it will will check  has_parent_in_container() ? 1 : 0
	//    so maximum size of overlapped pairs is  beam_width-1 

	std::vector<LookUpEntry<Type, IndexType>> lookUp;
	lookUp.resize(beam_width - 1);

	//additional storage to sort overlapped case by classes
	std::vector<std::pair<IndexType, int >> child_class_sorter_help;
	child_class_sorter_help.resize(beam_width - 1);
	

	for (uint64_t t = 0; t < len_t; t++)
	{
		auto next_beam_size = 0;
		//std::cout << "---" << t << "---" << std::endl;
		for (auto j = 0; j < last_beam_size; j++)
		{
			SequenceNode<IndexType>* seq = last_beams[j].entry.sequence;
			auto& cur_prob = last_beams[j].entry.prob;
			//if len(seq) > 0 then
			const auto log_p_blank = element<HasElementStride>(log_p, blank_index, element_stride);

			//log_p[seq->value] 
			auto non_blank_prob = seq->value != -1 ? (element<HasElementStride>(log_p, seq->value, element_stride) + cur_prob.non_blank) : negative_infinity<Type>();

			const Type blank_prob = log_p_blank + cur_prob.total;


			auto look_up_beam_index = -1;

			if (last_beams[j].index_as_child != -1)
			{
				//check entry
				look_up_beam_index = lookUp[last_beams[j].index_as_child].next_beam_index;
			}

			if (look_up_beam_index == -1)
			{
				BeamEntryType entry;
				entry.sequence = seq;
				entry.prob.blank = blank_prob;
				entry.prob.non_blank = non_blank_prob;
				entry.prob.total = log_sum_exp(blank_prob, non_blank_prob);
				next_beams[next_beam_size] = entry;
				//map if its overlapped one. in this case just being child is enough
				if (last_beams[j].index_as_child != -1)
				{
					lookUp[last_beams[j].index_as_child].next_beam_index = next_beam_size;
				}
				++next_beam_size;
			}
			else
			{
				//note: here we took as ref &
				auto& entry_prob = next_beams[look_up_beam_index].prob;
				entry_prob.blank = log_sum_exp(entry_prob.blank, blank_prob);
				entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
				entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.non_blank);
			}
			//check to see if it is overlapped parent and possible hits
			auto start_index = last_beams[j].index_as_parent;
			auto end_index = last_beams[j].index_as_parent + last_beams[j].children_count;

			for (int c = 0; c < len_c; c++)
			{
				if (c == blank_index) continue;

				const auto prob = element<HasElementStride>(log_p, c, element_stride);//log_p[c];

				non_blank_prob = pr(c, cur_prob, seq, prob);

				//extend by new character 
				auto look_up_beam_index_ex = -1; 
				int found_index = -1;

				//get index within array if its that class index
				if (start_index < end_index && lookUp[start_index].last_c == c){
						look_up_beam_index_ex = lookUp[start_index].next_beam_index;
						
						found_index = start_index;
						++start_index; 
				}

				if (look_up_beam_index_ex == -1)
				{
					BeamEntryType entry;
					SequenceNode<IndexType>* extended_sequence;
					if (found_index!=-1)
					{ 
						extended_sequence = lookUp[found_index].node;
						//assing next_beam_index for lookup
						lookUp[found_index].next_beam_index = next_beam_size;
						extended_sequence->increaseRef();
					}
					else {
						extended_sequence = sequence_container.extendPath(seq, c);
					}
					entry.prob.non_blank = non_blank_prob;
					entry.prob.total = non_blank_prob;
					entry.sequence = extended_sequence;
					next_beams[next_beam_size] = entry;

					++next_beam_size;
				}
				else
				{
					auto& entry_prob = next_beams[look_up_beam_index_ex].prob;
					entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
					entry_prob.total = log_sum_exp(entry_prob.total, non_blank_prob);
				}
			} //iteration over classes

			//mark it as extended
			seq->markAsFullyExtended();

		} //iteration over  beams

		log_p += inc_p;

#if 0
		std::cout << "---" << t << "---" << std::endl;
		//print
		for (int j = 0; j < next_beam_size; j++)
		{
			//std::cout << " in container " << (long long)next_beams[j].sequence << std::endl;
			print_seq1(next_beams[j].sequence);
			std::cout << std::setprecision(17) << next_beams[j].prob.total << std::endl;
		}
		std::cout << "\n\n\n" << std::endl;
#endif
		last_beam_size = std::min(next_beam_size, beam_width);
#if !defined(NTH_ELEMENT)
		//sort next beams to get candidates
		std::partial_sort(std::begin(next_beams),
			std::begin(next_beams) + last_beam_size,
			std::begin(next_beams) + next_beam_size, compare_beam_prob<Type, IndexType>);

#else
		std::nth_element(std::begin(next_beams),
			std::begin(next_beams) + last_beam_size,
			std::begin(next_beams) + next_beam_size, compare_beam_prob<Type, IndexType>);

#endif
#if 0
		std::cout << "--candidates-" << t << "---" << std::endl;
		//print
		for (int j = 0; j < last_beam_size; j++)
		{
			//std::cout << " in container " << (long long)next_beams[j].sequence << std::endl;
			print_seq1(next_beams[j].sequence);
			std::cout << std::setprecision(17) << next_beams[j].prob.total << std::endl;
		}
		std::cout << "\n\n\n" << std::endl;
#endif		

		if (t < len_t)
		{
			//copy top beams
			for (int j = 0; j < last_beam_size; j++)
			{
				last_beams[j].entry = next_beams[j];
				last_beams[j].index_as_child = -1;
				last_beams[j].index_as_parent = -1;
				last_beams[j].children_count = 0;
			}


			//delete sequences from the sequence_holder to decrease memory
			for (auto j = beam_width; j < next_beam_size; j++)
			{
				sequence_container.remove(next_beams[j].sequence);
			}


			//check overlap_direction and create lookUp with sorted classes as well
			//std::cout <<  "-childlookup-" << t<<"-----" << std::endl;
			int look_up_index = 0; 
			for (auto j = 0; j < last_beam_size; j++)
			{
				//if it is not parent node then there is not any need to check
				if (last_beams[j].entry.sequence->isFullyExtended())
				{
					auto parent_seq=last_beams[j].entry.sequence;
					int children_count = 0;
					for (int k = 0; k < last_beam_size; k++)
					{
						auto current = last_beams[k].entry.sequence;
						if (current->prefix == parent_seq)
						{ 
							child_class_sorter_help[children_count].first = current->value;
							child_class_sorter_help[children_count].second = k ;
							++children_count ;
						}
					}

					if (children_count > 0)
					{
						//std::cout << j << "-children-" << children_count << std::endl;
						//sort by class
						if(children_count<2){
							// 
							if (children_count > 1 && child_class_sorter_help[0].first > child_class_sorter_help[1].first)
							{
								std::swap(child_class_sorter_help[0], child_class_sorter_help[1]);
							}
						}
						else
						{
							std::sort(std::begin(child_class_sorter_help), std::begin(child_class_sorter_help) + children_count,
								[](const std::pair<int, int>& left, const std::pair<int, int>& right) {
									return left.first < right.first;
								});
						}
						last_beams[j].index_as_parent = look_up_index;
						last_beams[j].children_count = children_count;

						for (int l = 0; l < children_count; l++)
						{
							
							int c = child_class_sorter_help[l].first;
							int k = child_class_sorter_help[l].second;
							//std::cout << c <<" , " << k << std::endl;
							last_beams[k].index_as_child = look_up_index;
							auto seq = last_beams[k].entry.sequence;
							lookUp[look_up_index].last_c = c;
							lookUp[look_up_index].node = seq;
							lookUp[look_up_index].next_beam_index = -1;
							//next one
							++look_up_index;
						}
					}//add sorted lookUps

				}
			} //overlap_direction identified to speed up lookUp
			
		}
		
	}//iterate over t
#if defined(NTH_ELEMENT)
	//use sort  for n elements as only nth_element was used
	std::sort(std::begin(next_beams), std::begin(next_beams) + last_beam_size, compare_beam_prob<Type, IndexType>);
#endif
	//store nbest results
	if (nbest_len <= last_beam_size) {
		for (int j = 0; j < nbest_len; j++)
		{
			auto top = next_beams[j];
			auto result_vector = SequenceContainer<IndexType>::getSequence(top.sequence, len_t);
			const auto seq_size = result_vector.size();

			result_prob[j] = top.prob.total;
			result_seq_length[j] = seq_size;
			//copy sequence
			for (auto s = 0; s < seq_size; s++)
			{
				result_sequence[s] = result_vector[s];
			}

			result_sequence += inc_res_seq;

		}
	}
	else
	{
		for (int j = 0; j < nbest_len; j++)
		{
			result_prob[j] = negative_infinity<Type>();
			result_seq_length[j] = 0;;
		}
	}
	return;
}



 
template<typename Type, typename IndexType = int>
void
beamSearch(const NDArray& logit, const NDArray& sequence_length, NDArray& result_sequences, NDArray& result_probs, NDArray& result_sequences_length, int blank_index, int beam_width = 25, int nbest_len = 1)
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
	assert(result_probs.ews() == 1 && result_probs.rankOf() == 2 && result_probs.shapeOf()[0] == batch_len && result_probs.shapeOf()[1] == nbest_len);
	//result sequence should be [batch_len, nbest_len,  max_len_t]
	assert(result_sequences.ews() == 1 && result_sequences.rankOf() == 3 && result_sequences.shapeOf()[0] == batch_len && result_sequences.shapeOf()[1] == nbest_len && result_sequences.shapeOf()[2] == max_len_t);

	auto result_seq_ptr = result_sequences.bufferAsT<IndexType>();
	auto result_probs_ptr = result_probs.bufferAsT<Type>();
	auto result_seq_length_ptr = result_sequences_length.bufferAsT<IndexType>();

	const auto  batch_stride_res = result_sequences.stridesOf()[0];
	const auto  inc_res = result_sequences.stridesOf()[1];
	const auto  batch_stride_res_prob = result_probs.stridesOf()[0];
	const auto  batch_stride_res_seq_length = result_sequences_length.stridesOf()[0];
	auto func = [max_len_t, len_c, batch_stride, inc_p, element_stride, element_stride_t, logits_ptr, len_t_ptr, blank_index, beam_width,
		nbest_len, result_seq_ptr, result_seq_length_ptr, result_probs_ptr, batch_stride_res, inc_res, batch_stride_res_prob, batch_stride_res_seq_length]
		(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void
	{

		auto ptr = logits_ptr + start * batch_stride;

		if (element_stride == 1)
		{
			//choose ews one
			for (auto b = start; b < stop; b += increment)
			{
				auto prob_ptr = &(result_probs_ptr[b * batch_stride_res_prob]);
				auto seq_length_ptr = &(result_seq_length_ptr[b * batch_stride_res_seq_length]);
				auto seq_ptr = &(result_seq_ptr[b * batch_stride_res]);

				auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
				inner_beam_search<false, Type, IndexType>(ptr, inc_p, seq_ptr, inc_res, max_len_t, prob_ptr, seq_length_ptr, len_t, len_c, blank_index, beam_width, nbest_len);

				ptr += batch_stride;

			}
		}
		else
		{
			// element with stride case 
			for (auto b = start; b < stop; b += increment)
			{
				auto prob_ptr = &(result_probs_ptr[b * batch_stride_res_prob]);
				auto seq_length_ptr = &(result_seq_length_ptr[b * batch_stride_res_seq_length]);
				auto seq_ptr = &(result_seq_ptr[b * batch_stride_res]);

				auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
				inner_beam_search<false, Type, IndexType>(ptr, inc_p, seq_ptr, inc_res, max_len_t, prob_ptr, seq_length_ptr, len_t, len_c, blank_index, beam_width, nbest_len, element_stride);

				ptr += batch_stride;
			}
		}
	};
	samediff::Threads::parallel_for(func, 0, batch_len, 1);
	return;
}



template<bool HasElementStride = false, typename Type, typename IndexType>
void inner_beam_search_(const Type* log_p, const uint64_t inc_p, IndexType* result_sequence, const uint64_t inc_res_seq,
	const uint64_t max_len_t, Type* result_prob, IndexType* result_seq_length, uint64_t len_t,
	const uint64_t len_c, const int blank_index, int beam_width, int nbest_len, bool normalize_logits, const uint64_t element_stride = 1L)
{

	using BeamEntryType = BeamEntry<Type, IndexType>;
	using BeamEntryTypeEx = BeamEntryEx<Type, IndexType>;

	if (beam_width < 1) beam_width = 1;
	if (nbest_len > beam_width) nbest_len = beam_width;
	//if len_t is greater than max_len_t truncate it
	len_t = len_t > max_len_t ? max_len_t : len_t;

	SequenceContainer<IndexType> sequence_container;
	BeamEntryType empty;
	empty.prob.blank = 0;
	empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.non_blank);
	empty.sequence = sequence_container.getEmptyPath();

	//vectors: we will use it as array, here
	std::vector<BeamEntryTypeEx> last_beams;
	std::vector<BeamEntryType> next_beams;
	last_beams.resize(beam_width);
	//as we skip blank indexes the count is beam_width * len_c 
	next_beams.resize(beam_width * len_c);
	last_beams[0] = { empty, -1, -1, 0 };
	auto last_beam_size = 1;

	// lookupContainer:
	// it will keep sorted entries. so we will just move and compare the entry
	// in each step there will be overlapped cases
	// the size of overlapped cases in last_beam[0:beam_width]:
	//    as we have beam_width size in each step after sort and pruning
	//    there is at least one item who will not have any parent
	//    and for the rest (beam_width-1) it will will check  has_parent_in_container() ? 1 : 0
	//    so maximum size of overlapped pairs is  beam_width-1 

	std::vector<LookUpEntry<Type, IndexType>> lookUp;
	lookUp.resize(beam_width - 1);

	//additional storage to sort overlapped case by classes
	std::vector<std::pair<IndexType, int >> child_class_sorter_help;
	child_class_sorter_help.resize(beam_width - 1);
	Type norm_offset = 0;

	for (uint64_t t = 0; t < len_t; t++)
	{
		auto next_beam_size = 0;
		if (normalize_logits) {
			norm_offset = softmax_normalization_term<HasElementStride, Type, IndexType>(log_p, len_c, element_stride);
		}
		for (auto j = 0; j < last_beam_size; j++)
		{
			SequenceNode<IndexType>* seq = last_beams[j].entry.sequence;
			auto& cur_prob = last_beams[j].entry.prob;
			//if len(seq) > 0 then
			const auto log_p_blank = element<HasElementStride>(log_p, blank_index, element_stride);
			Type blank_prob, non_blank_prob;
			non_blank_prob = seq->value != -1 ? (element<HasElementStride>(log_p, seq->value, element_stride) + cur_prob.non_blank) : negative_infinity<Type>();
			blank_prob = log_p_blank + cur_prob.total;

			if (normalize_logits) {
				
				non_blank_prob = non_blank_prob - norm_offset;
				blank_prob = blank_prob - norm_offset;
			}

			auto look_up_beam_index = -1;

			if (last_beams[j].index_as_child != -1)
			{
				//check entry
				look_up_beam_index = lookUp[last_beams[j].index_as_child].next_beam_index;
			}

			if (look_up_beam_index == -1)
			{
				BeamEntryType entry;
				entry.sequence = seq;
				entry.prob.blank = blank_prob;
				entry.prob.non_blank = non_blank_prob;
				entry.prob.total = log_sum_exp(blank_prob, non_blank_prob);
				next_beams[next_beam_size] = entry;
				//map if its overlapped one. in this case just being child is enough
				if (last_beams[j].index_as_child != -1)
				{
					lookUp[last_beams[j].index_as_child].next_beam_index = next_beam_size;
				}
				++next_beam_size;
			}
			else
			{
				//note: here we took as ref &
				auto& entry_prob = next_beams[look_up_beam_index].prob;
				entry_prob.blank = log_sum_exp(entry_prob.blank, blank_prob);
				entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
				entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.non_blank);
			}
			//check to see if it is overlapped parent
			auto start_index = last_beams[j].index_as_parent;
			auto end_index = last_beams[j].index_as_parent + last_beams[j].children_count;

			for (int c = 0; c < len_c; c++)
			{
				if (c == blank_index) continue;

				const auto prob = element<HasElementStride>(log_p, c, element_stride);//log_p[c];

				non_blank_prob = pr(c, cur_prob, seq, prob);
				if (normalize_logits) non_blank_prob = non_blank_prob - norm_offset;
				//extend by new character 
				auto look_up_beam_index_ex = -1;
				int found_index = -1;

				//get index within array if its that class index
				if (start_index < end_index && lookUp[start_index].last_c == c) {
					look_up_beam_index_ex = lookUp[start_index].next_beam_index;

					found_index = start_index;
					++start_index;
				}

				if (look_up_beam_index_ex == -1)
				{
					BeamEntryType entry;
					SequenceNode<IndexType>* extended_sequence;
					if (found_index != -1)
					{
						extended_sequence = lookUp[found_index].node;
						//assing next_beam_index for lookup
						lookUp[found_index].next_beam_index = next_beam_size;
						extended_sequence->increaseRef();
					}
					else {
						extended_sequence = sequence_container.extendPath(seq, c);
					}
					entry.prob.non_blank = non_blank_prob;
					entry.prob.total = non_blank_prob;
					entry.sequence = extended_sequence;
					next_beams[next_beam_size] = entry;

					++next_beam_size;
				}
				else
				{
					auto& entry_prob = next_beams[look_up_beam_index_ex].prob;
					entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
					entry_prob.total = log_sum_exp(entry_prob.total, non_blank_prob);
				}
			} //iteration over classes

			//mark it as extended
			seq->markAsFullyExtended();

		} //iteration over  beams

		log_p += inc_p;

		last_beam_size = std::min(next_beam_size, beam_width);
#if !defined(NTH_ELEMENT)
		//sort next beams to get candidates
		std::partial_sort(std::begin(next_beams),
			std::begin(next_beams) + last_beam_size,
			std::begin(next_beams) + next_beam_size, compare_beam_prob<Type, IndexType>);

#else
		std::nth_element(std::begin(next_beams),
			std::begin(next_beams) + last_beam_size,
			std::begin(next_beams) + next_beam_size, compare_beam_prob<Type, IndexType>);

#endif    

		if (t < len_t)
		{
			//copy top beams
			for (int j = 0; j < last_beam_size; j++)
			{
				last_beams[j].entry = next_beams[j];
				last_beams[j].index_as_child = -1;
				last_beams[j].index_as_parent = -1;
				last_beams[j].children_count = 0;
			}


			//delete sequences from the sequence_holder to decrease memory
			for (auto j = beam_width; j < next_beam_size; j++)
			{
				sequence_container.remove(next_beams[j].sequence);
			}


			//check overlapping cases and create lookUp with sorted classes as well
			int look_up_index = 0;
			for (auto j = 0; j < last_beam_size; j++)
			{
				//if it is not parent node then there is not any need to check
				if (last_beams[j].entry.sequence->isFullyExtended())
				{
					auto parent_seq = last_beams[j].entry.sequence;
					int children_count = 0;
					for (int k = 0; k < last_beam_size; k++)
					{
						auto current = last_beams[k].entry.sequence;
						if (current->prefix == parent_seq)
						{
							child_class_sorter_help[children_count].first = current->value;
							child_class_sorter_help[children_count].second = k;
							++children_count;
						}
					}

					if (children_count > 0)
					{

						//sort by class
						if (children_count < 2) {
							// 
							if (children_count > 1 && child_class_sorter_help[0].first > child_class_sorter_help[1].first)
							{
								std::swap(child_class_sorter_help[0], child_class_sorter_help[1]);
							}
						}
						else
						{
							std::sort(std::begin(child_class_sorter_help), std::begin(child_class_sorter_help) + children_count,
								[](const std::pair<int, int>& left, const std::pair<int, int>& right) {
									return left.first < right.first;
								});
						}
						last_beams[j].index_as_parent = look_up_index;
						last_beams[j].children_count = children_count;

						for (int l = 0; l < children_count; l++)
						{

							int c = child_class_sorter_help[l].first;
							int k = child_class_sorter_help[l].second;
							//std::cout << c <<" , " << k << std::endl;
							last_beams[k].index_as_child = look_up_index;
							auto seq = last_beams[k].entry.sequence;
							lookUp[look_up_index].last_c = c;
							lookUp[look_up_index].node = seq;
							lookUp[look_up_index].next_beam_index = -1;
							//next one
							++look_up_index;
						}
					}//add sorted lookUps

				}
			} //overlap_direction identified to speed up lookUp

		}

	}//iterate over t
#if defined(NTH_ELEMENT)
	//use sort  for n elements as only nth_element was used
	std::sort(std::begin(next_beams), std::begin(next_beams) + last_beam_size, compare_beam_prob<Type, IndexType>);
#endif
	//store nbest results
	if (nbest_len <= last_beam_size) {
		for (int j = 0; j < nbest_len; j++)
		{
			auto top = next_beams[j];
			auto result_vector = SequenceContainer<IndexType>::getSequence(top.sequence, len_t);
			const auto seq_size = result_vector.size();

			result_prob[j] = top.prob.total;
			result_seq_length[j] = seq_size;
			//copy sequence
			for (auto s = 0; s < seq_size; s++)
			{
				result_sequence[s] = result_vector[s];
			}

			result_sequence += inc_res_seq;

		}
	}
	else
	{
		for (int j = 0; j < nbest_len; j++)
		{
			result_prob[j] = negative_infinity<Type>();
			result_seq_length[j] = 0;;
		}
	}
	return;
}

template<typename Type, typename IndexType = int>
void
beamSearch_(const NDArray& logit, const NDArray& sequence_length, NDArray& result_sequences, NDArray& result_probs, NDArray& result_sequences_length, int blank_index, int beam_width, int nbest_len, bool normalize_logits = true)
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
	assert(result_probs.ews() == 1 && result_probs.rankOf() == 2 && result_probs.shapeOf()[0] == batch_len && result_probs.shapeOf()[1] == nbest_len);
	//result sequence should be [batch_len, nbest_len,  max_len_t]
	assert(result_sequences.ews() == 1 && result_sequences.rankOf() == 3 && result_sequences.shapeOf()[0] == batch_len && result_sequences.shapeOf()[1] == nbest_len && result_sequences.shapeOf()[2] == max_len_t);

	auto result_seq_ptr = result_sequences.bufferAsT<IndexType>();
	auto result_probs_ptr = result_probs.bufferAsT<Type>();
	auto result_seq_length_ptr = result_sequences_length.bufferAsT<IndexType>();

	const auto  batch_stride_res = result_sequences.stridesOf()[0];
	const auto  inc_res = result_sequences.stridesOf()[1];
	const auto  batch_stride_res_prob = result_probs.stridesOf()[0];
	const auto  batch_stride_res_seq_length = result_sequences_length.stridesOf()[0];
	auto func = [max_len_t, len_c, batch_stride, inc_p, element_stride, element_stride_t, logits_ptr, len_t_ptr, blank_index, beam_width, normalize_logits,
		nbest_len, result_seq_ptr, result_seq_length_ptr, result_probs_ptr, batch_stride_res, inc_res, batch_stride_res_prob, batch_stride_res_seq_length]
		(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void
	{

		auto ptr = logits_ptr + start * batch_stride;

		if (element_stride == 1)
		{
			//choose ews one
			for (auto b = start; b < stop; b += increment)
			{
				auto prob_ptr = &(result_probs_ptr[b * batch_stride_res_prob]);
				auto seq_length_ptr = &(result_seq_length_ptr[b * batch_stride_res_seq_length]);
				auto seq_ptr = &(result_seq_ptr[b * batch_stride_res]);

				auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
				inner_beam_search_<false, Type, IndexType>(ptr, inc_p, seq_ptr, inc_res, max_len_t, prob_ptr, seq_length_ptr, len_t, len_c, blank_index, beam_width, nbest_len, normalize_logits);

				ptr += batch_stride;

			}
		}
		else
		{
			// element with stride case 
			for (auto b = start; b < stop; b += increment)
			{
				auto prob_ptr = &(result_probs_ptr[b * batch_stride_res_prob]);
				auto seq_length_ptr = &(result_seq_length_ptr[b * batch_stride_res_seq_length]);
				auto seq_ptr = &(result_seq_ptr[b * batch_stride_res]);

				auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
				inner_beam_search_<false, Type, IndexType>(ptr, inc_p, seq_ptr, inc_res, max_len_t, prob_ptr, seq_length_ptr, len_t, len_c, blank_index, beam_width, nbest_len, normalize_logits, element_stride);

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
	auto result_seq_length = NDArrayFactory::create<int>('c', { batch_len, nbest });
	auto result_seq = NDArrayFactory::create<int>('c', { batch_len , nbest, max_len_t });

	op(logInput, seq_len, result_seq, result_prob, result_seq_length, blankInd, w, nbest, true);

	result_seq_length.printIndexedBuffer("sequence length");
	result_seq.printIndexedBuffer("result seq");
	result_prob.printIndexedBuffer("result prob");

	auto res_ptr = result_seq.bufferAsT<int>();
	auto result_seq_length_ptr = result_seq_length.bufferAsT<int>();
	const auto b_size = result_seq_length.stridesOf()[0];
	const auto inc = result_seq.stridesOf()[1];
	//print one of seq as lchars
	for (int j = 0; j < nbest; j++) {
		for (int i = 0; i < result_seq_length_ptr[0]; i++) {
			std::cout << classes[res_ptr[i]] << ",";
		}
		std::cout << std::endl;
		res_ptr += inc;
	}
}


constexpr int global_beam_width = 25;

extern "C" float test_new_improved()
{
	static NDArray logInput, seq_length, result_prob, result_seq, result_seq_length;
	static int t = 0;
	const std::string classes = " !\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'";
	if (t == 0)
	{
		const std::vector<float> data = load_file("rnn.csv");
		logInput = NDArrayFactory::create<float>('c', { 1, 100, 80 }, data);
		seq_length = NDArrayFactory::create<int>('c', { 1 }, { 100 });
		result_prob = NDArrayFactory::create<float>('c', { 1, 1 });
		result_seq = NDArrayFactory::create<int>('c', { 1 , 1, 100 });
		result_seq_length = NDArrayFactory::create<int>('c', { 1, 1 });
		softmax<float, 3>(logInput, true);
		t = 1;
	}


	beamSearch_<float, int>(logInput, seq_length, result_seq, result_prob, result_seq_length, logInput.shapeOf()[1] - 1, global_beam_width, 1);
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
	auto result_seq_len = NDArrayFactory::create<int>('c', { BATCH_LEN , NBEST_LEN });
	beamSearch_<float, int>(logits, logits_length, result_seq, result_prob, result_seq_len, BLANK_INDEX, BEAM_WIDTH, NBEST_LEN);

	result_seq_len.printIndexedBuffer("seq length");
	result_seq.printIndexedBuffer("result seq");
	result_prob.printIndexedBuffer("result prob");
}



int main(int argc, char** argv)
{

	test123();

	// int yyy; std::cin >> yyy;
	//return -1;
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

	testb(beamSearch_<float, int>, logInput3, seq3, classes, 3, 2);

	testb(beamSearch_<float, int>, logInputB, seq_b, classes, global_beam_width, 4);

	//testb(beamSearch_<float, int>, logInputB, seq_b2, classes, 25, 4);
#if 1
	auto beam_width = global_beam_width;

	auto batch_len = logInputB.rankOf() > 2 ? logInputB.shapeOf()[0] : 1;
	auto result_prob = NDArrayFactory::create<float>('c', { batch_len, 1 });
	auto result_seq = NDArrayFactory::create<int>('c', { batch_len , 1, 100 });
	auto result_seq_len = NDArrayFactory::create<int>('c', { batch_len , 1 });
	beamSearch_<float, int>(logInputB, seq_b, result_seq, result_prob, result_seq_len, logInputB.shapeOf()[2] - 1, beam_width, 1);
#if  1
	time_it<10, 100>(beamSearch_<float, int>, 0, logInputB, seq_b, result_seq, result_prob, result_seq_len, logInputB.shapeOf()[2] - 1, beam_width, 1, true);
#endif

#if  1
	time_it<10, 100>(beamSearch_<float, int>, 0, logInputB, seq_b, result_seq, result_prob, result_seq_len, logInputB.shapeOf()[2] - 1, beam_width, 1, true);
#endif

#if  1
	time_it<10, 100>(beamSearch_<float, int>, 0, logInputB, seq_b, result_seq, result_prob, result_seq_len, logInputB.shapeOf()[2] - 1, beam_width, 1, false);
#endif

#if  1
	time_it<10, 100>(beamSearch_<float, int>, 0, logInputB, seq_b, result_seq, result_prob, result_seq_len, logInputB.shapeOf()[2] - 1, beam_width, 1, false);
#endif
#endif

	return -1;
}
