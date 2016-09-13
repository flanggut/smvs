/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_THREAD_POOL_HEADER
#define SMVS_THREAD_POOL_HEADER

#include <mutex>
#include <queue>
#include <thread>
#include <future>
#include <vector>
#include <condition_variable>

/*
 * C++11 Thread Pool
 *
 * based on https://github.com/progschj/ThreadPool
 */


/* Example usage:

    ThreadPool threadpool(4);

    std::vector< std::future<int> > results;

    for(int i = 0; i < 8; ++i) {
        results.emplace_back(
            pool.add_task([i] {
                return i*i;
            })
        );
    }

    for(auto && result: results)
        std::cout << result.get() << ' ';
    std::cout << std::endl;

*/

class ThreadPool
{
public:
    /* Constructor creating specified number of threads */
    explicit ThreadPool (std::size_t num_threads);
    ~ThreadPool (void);

    /* add task function to queue */
    template<class F, class... Args>
    auto add_task (F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    /* Worker threads */
    std::vector<std::thread> workers;

    /* Task queue */
    std::queue<std::function<void()>> tasks;

    /* Mutex for concurrent queue access */
    std::mutex tasks_mutex;

    /* Condition variable to wake idle threads */
    std::condition_variable condition;

    /* Thread pool has been stopped */
    bool stop;
};

/* ---------------------------------------------------------------- */

inline
ThreadPool::ThreadPool(std::size_t num_threads)
    : stop(false)
{
    for (std::size_t i = 0; i < num_threads; ++i)
    {
        this->workers.emplace_back([this,i]{
            while (true)
            {
                /* acquire current task from queue */
                std::function<void()> current_task;
                {
                    /* acquire lock on queue */
                    std::unique_lock<std::mutex> lock(this->tasks_mutex);

                    /* wait only if queue is empty and still running */
                    this->condition.wait(lock, [this]{
                        return (this->stop || !this->tasks.empty());
                    });

                    /* stop thread if queue is finished */
                    if(this->stop && this->tasks.empty())
                        return;

                    /* get task from front of queue */
                    current_task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                /* run current task */
                current_task();
            }
        });
    }
}

inline
ThreadPool::~ThreadPool ()
{
    /* acquire queue lock */
    {
        std::lock_guard<std::mutex> lock(tasks_mutex);
        this->stop = true;
    }
    /* wake all waiting threads */
    this->condition.notify_all();

    /* join all worker threads */
    for (std::thread & worker : this->workers)
        worker.join();
}



template<class F, class... Args>
auto ThreadPool::add_task(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    /* get return type of function */
    using return_type = typename std::result_of<F(Args...)>::type;

    /* create new task */
    auto task = std::make_shared<std::packaged_task<return_type()>> (
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    /* get future return value from task */
    std::future<return_type> res = task->get_future();
    {
        /* acquire queue lock */
        std::lock_guard<std::mutex> lock(tasks_mutex);

        if(stop)
            throw std::runtime_error("ThreadPool has been destroyed"
             " cannot add more tasks");

        /* enqueue new task */
        tasks.emplace([task](){ (*task)(); });
    }
    /* wake one waiting thread to perform work */
    condition.notify_one();

    /* return future result */
    return res;
}

#endif /* SMVS_THREAD_POOL_HEADER */
