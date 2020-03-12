/*******************************************************************************
 * Copyright (c) 2019 Konduit
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

 //
 // @author raver119@gmail.com
 //
#ifndef SAMEDIFF_THREADS_H
#define SAMEDIFF_THREADS_H

#include <functional>
#include <NDX.h>
#include <thread>
#include   <mutex> 
#include <cstddef>
#include <type_traits>
#include <utility>
#include <Environment.h>
 
namespace samediff {
    class ND4J_EXPORT ThreadsHelper { 
    public:
        static int numberOfThreads(int maxThreads, uint64_t numberOfElements);
        static int numberOfThreads2d(int maxThreads, uint64_t iters_x, uint64_t iters_y);
        static int numberOfThreads3d(int maxThreads, uint64_t iters_x, uint64_t iters_y, uint64_t iters_z);
        static int pickLoop2d(int numThreads, uint64_t iters_x, uint64_t iters_y);
        static int pickLoop3d(int numThreads, uint64_t iters_x, uint64_t iters_y, uint64_t iters_z);
    };

    class ND4J_EXPORT Span {
    private:
        int64_t _startX, _stopX, _incX;
    public:
        Span(int64_t start_x, int64_t stop_x, int64_t inc_x);
        ~Span() = default;

        int64_t startX() const;
        int64_t stopX() const;
        int64_t incX() const;

        static Span build(uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x);
    };

    class ND4J_EXPORT Span2 {
    private:
        int64_t _startX, _stopX, _incX;
        int64_t _startY, _stopY, _incY;
    public:
        Span2(int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y);
        ~Span2() = default;

        int64_t startX() const;
        int64_t startY() const;

        int64_t stopX() const;
        int64_t stopY() const;

        int64_t incX() const;
        int64_t incY() const;

        static Span2 build(int loop, uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y);
    };

    class ND4J_EXPORT Span3 {
    private:
        int64_t _startX, _stopX, _incX;
        int64_t _startY, _stopY, _incY;
        int64_t _startZ, _stopZ, _incZ;
    public:
        Span3(int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z);
        ~Span3() = default;

        int64_t startX() const;
        int64_t startY() const;
        int64_t startZ() const;

        int64_t stopX() const;
        int64_t stopY() const;
        int64_t stopZ() const;

        int64_t incX() const;
        int64_t incY() const;
        int64_t incZ() const;

        static Span3 build(int loop, uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z);
    };

    class ND4J_EXPORT Threads {
 
    public:
    static std::mutex gThreadmutex;
	static uint64_t _nFreeThreads; 
    static bool   tryAcquire(int numThreads);
    static bool   freeThreads(int numThreads);
  

    public:
        /**
         * This function executes 1 dimensional loop for a given number of threads
         * PLEASE NOTE: this function can use smaller number of threads than requested.
         *
         * @param function
         * @param numThreads
         * @param start
         * @param stop
         * @param increment
         * @return
         */
        static int parallel_for(FUNC_1D function, int64_t start, int64_t stop, int64_t increment = 1, uint32_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        /**
         * This function executes 1 dimensional loop for a given number of threads
         *
         * @param function
         * @param start
         * @param stop
         * @param increment
         * @param numThreads
         * @return
         */
        static int parallel_tad(FUNC_1D function, int64_t start, int64_t stop, int64_t increment = 1, uint32_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        /**
         * This method will execute function splitting 2 nested loops space with multiple threads
         *
         * @param function
         * @param numThreads
         * @param start_x
         * @param stop_x
         * @param inc_x
         * @param start_y
         * @param stop_y
         * @param inc_y
         * @return
         */
        static int parallel_for(FUNC_2D function, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads(), bool debug = false);

        /**
         * This method will execute function splitting 3 nested loops space with multiple threads
         *
         * @param function
         * @param numThreads
         * @param start_x
         * @param stop_x
         * @param inc_x
         * @param start_y
         * @param stop_y
         * @param inc_y
         * @param start_z
         * @param stop_z
         * @param inc_z
         * @return
         */
        static int parallel_for(FUNC_3D function, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        /**
         *
         * @param function
         * @param numThreads
         * @return
         */
        static int parallel_do(FUNC_DO function, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        static int64_t parallel_long(FUNC_RL function, FUNC_AL aggregator, int64_t start, int64_t stop, int64_t increment = 1, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        static double parallel_double(FUNC_RD function, FUNC_AD aggregator, int64_t start, int64_t stop, int64_t increment = 1, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        /**
         * This method will execute function in parallel preserving the parts to be aligned increment size
         * PLEASE NOTE: this function can use smaller number of threads than requested.
         *
        */
        static int  parallel_aligned_increment(FUNC_1D function, int64_t start, int64_t stop, int64_t increment, bool adjust = true,size_t type_size = sizeof(float), uint32_t req_numThreads = sd::Environment::getInstance()->maxMasterThreads());


        template<typename Op, typename... Args >
        static int   parallel_aligned_increment2(Op op, int64_t start, int64_t stop, int64_t increment, Args&&... args) {
            if (start > stop)
                throw std::runtime_error("Threads::parallel_for got start > stop");
            auto num_elements = (stop - start);
            //this way we preserve increment starts offset
            //so we will parition considering delta but not total elements
            auto delta = (stop - start) / increment;
            int req_numThreads = 8;
            int type_size = 1;

            // in some cases we just fire func as is
            if (delta == 0 || req_numThreads == 1) {
                op(0, start, stop, increment, std::forward<Args>(args)...);
                return 1;
            }
            int numThreads = 0;

            struct th_span {
                Nd4jLong start;
                Nd4jLong end;
            };

#ifdef __NEC__
            constexpr int max_thread_count = 8;
#else
            constexpr int max_thread_count = 1024;
#endif
            th_span thread_spans[max_thread_count];

            req_numThreads = req_numThreads > max_thread_count ? max_thread_count : req_numThreads;

#ifdef __NEC__
            int adjusted_numThreads = max_thread_count;
#else
            int adjusted_numThreads = (!false) ? req_numThreads : samediff::ThreadsHelper::numberOfThreads(req_numThreads, (num_elements * sizeof(double)) / (200 * type_size));
#endif

            if (adjusted_numThreads > delta)
                adjusted_numThreads = delta;
            // shortcut
            if (adjusted_numThreads <= 1) {
                op(0, start, stop, increment, std::forward<Args>(args)...);
                return 1;
            }



            //take span as ceil
            auto spand = std::ceil((double)delta / (double)adjusted_numThreads);
            numThreads = static_cast<int>(std::ceil((double)delta / spand));
            auto span = static_cast<Nd4jLong>(spand);


            //tail_add is additional value of the last part
            //it could be negative or positive
            //we will spread that value across
            auto tail_add = delta - numThreads * span;
            Nd4jLong begin = 0;
            Nd4jLong end = 0;

            //we will try enqueu bigger parts first
            decltype(span) span1, span2;
            int last = 0;
            if (tail_add >= 0) {
                //for span == 1  , tail_add is  0
                last = tail_add;
                span1 = span + 1;
                span2 = span;
            }
            else {
                last = numThreads + tail_add;// -std::abs(tail_add);
                span1 = span;
                span2 = span - 1;
            }
            for (int i = 0; i < last; i++) {
                end = begin + span1 * increment;
                // putting the task into the queue for a given thread
                thread_spans[i].start = begin;
                thread_spans[i].end = end;
                begin = end;
            }
            for (int i = last; i < numThreads - 1; i++) {
                end = begin + span2 * increment;
                // putting the task into the queue for a given thread
                thread_spans[i].start = begin;
                thread_spans[i].end = end;
                begin = end;
            }
            //for last one enqueue last offset as stop
            //we need it in case our ((stop-start) % increment ) > 0
            thread_spans[numThreads - 1].start = begin;
            thread_spans[numThreads - 1].end = stop;
#if 0
            nd4j_printf("use#1 %d free %d\n", numThreads, _nFreeThreads);
#endif
            if (true /*tryAcquire(numThreads)*/) {
#if 0
                nd4j_printf("use#2 %d free %d\n", numThreads, _nFreeThreads);
#endif
#if 1
#pragma omp parallel for
#else
#pragma _NEC parallel
#endif
                for (size_t j = 0; j < numThreads; j++) {
#if 0
                    nd4j_printf("use#inner span %ld   %ld\n", thread_spans[j].start, thread_spans[j].end);
#endif
                    op(j, thread_spans[j].start, thread_spans[j].end, increment, std::forward<Args>(args)...);
                }
               // freeThreads(numThreads);
                return numThreads;
            }
            else {
#if 0
                nd4j_printf("use#3 %d free %d\n", 1, _nFreeThreads);
#endif
                op(0, start, stop, increment, std::forward<Args>(args)...);
                // we tell that parallelism request declined
                return 1;
            }
        }
    };
}


#endif //SAMEDIFF_THREADS_H
