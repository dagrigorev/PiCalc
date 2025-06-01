#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <mutex>

// Указываем точность вычисления — в терминах числа членов ряда
constexpr int64_t NUM_TERMS = 15'000'000'000; // Если позволяет охлаждение !!!

// Получаем количество потоков (но гарантируем хотя бы 1)
const unsigned int NUM_THREADS = std::max(1u, std::thread::hardware_concurrency());

/// Вычисление одного члена BBP-формулы
double bbp_term(int64_t k) {
    double k8 = 8.0 * k;
    return (1.0 / std::pow(16.0, k)) * (
        4.0 / (k8 + 1.0) -
        2.0 / (k8 + 4.0) -
        1.0 / (k8 + 5.0) -
        1.0 / (k8 + 6.0)
        );
}

// Мьютекс вывода
std::mutex cout_mutex;

/// Параллельное вычисление Pi
double compute_pi_parallel() {
    std::vector<std::thread> threads;
    std::vector<double> partial_results(NUM_THREADS, 0.0);

    int64_t terms_per_thread = NUM_TERMS / NUM_THREADS;

    for (unsigned int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([t, terms_per_thread, &partial_results]() {
            int64_t start = t * terms_per_thread;
            int64_t end = (t == NUM_THREADS - 1) ? NUM_TERMS : start + terms_per_thread;
            double sum = 0.0;

            for (int64_t k = start; k < end; ++k) {
                sum += bbp_term(k);
#if DEBUG
                // Прогресс-бар: выводим каждую 1/10 часть блока потока
                if ((k - start) % (terms_per_thread / 10 + 1) == 0) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "Thread #" << t
                        << " | k = " << k
                        << '\n';
                }
#endif
            }

            partial_results[t] = sum;
#if DEBUG
            // Финальное сообщение от потока
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "Thread #" << t << " finished with partial Pi: "
                    << std::setprecision(23) << sum << "\n";
            }
#endif
            });
    }

    for (auto& th : threads)
        th.join();

    double pi = 0.0;
    for (const auto& r : partial_results)
        pi += r;

    return pi;
}

int main() {
    std::cout << "CPU Threads detected: " << NUM_THREADS << '\n';
    std::cout << "Calculating Pi using BBP formula with " << NUM_TERMS << " terms...\n";

    auto start = std::chrono::high_resolution_clock::now();
    double pi = compute_pi_parallel();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << std::setprecision(20) << "Computed Pi: " << pi << '\n';
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    return 0;
}

/*
 * // Intel(R) Core(TM) i7-14700K специфическая версия
 
// Быстрое возведение в степень 16^k для целого k
inline double pow16(int64_t k) {
    double result = 1.0;
    for (int64_t i = 0; i < k; ++i) result *= 16.0;
    return result;
}

// Вычисляет 4 BBP-слагаемых одновременно: k, k+1, k+2, k+3
inline __m256d bbp_term_vec4(int64_t base_k) {
    __m256d k = _mm256_set_pd(base_k + 3, base_k + 2, base_k + 1, base_k);

    __m256d k8 = _mm256_mul_pd(k, _mm256_set1_pd(8.0));

    __m256d pow16k = _mm256_set_pd(
        pow16(base_k + 3),
        pow16(base_k + 2),
        pow16(base_k + 1),
        pow16(base_k)
    );

    __m256d term = _mm256_sub_pd(
        _mm256_sub_pd(
            _mm256_div_pd(_mm256_set1_pd(4.0), _mm256_add_pd(k8, _mm256_set1_pd(1.0))),
            _mm256_div_pd(_mm256_set1_pd(2.0), _mm256_add_pd(k8, _mm256_set1_pd(4.0)))
        ),
        _mm256_add_pd(
            _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_add_pd(k8, _mm256_set1_pd(5.0))),
            _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_add_pd(k8, _mm256_set1_pd(6.0)))
        )
    );

    return _mm256_div_pd(term, pow16k);
}

// Суммирует значения в __m256d
inline double horizontal_add(__m256d v) {
    alignas(32) double buf[4];
    _mm256_store_pd(buf, v);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

// Глобальный атомарный прогресс (для визуализации)
std::atomic<int64_t> global_progress = 0;

void compute_chunk(int thread_id, int64_t start, int64_t end, double& result) {
    double sum = 0.0;
    for (int64_t k = start; k < end; k += 4) {
        __m256d vec = bbp_term_vec4(k);
        sum += horizontal_add(vec);

        // Показываем прогресс
        int64_t current = global_progress.fetch_add(4);
        if ((current / 1000000) != ((current + 4) / 1000000)) {
            std::cout << "[Thread " << thread_id << "] computed terms: " << (current + 4) << ", partial sum: " << sum << '\n';
        }
    }
    result = sum;
}

double compute_pi_parallel() {
    std::vector<std::thread> threads;
    std::vector<double> results(NUM_THREADS, 0.0);

    int64_t terms_per_thread = NUM_TERMS / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; ++t) {
        int64_t start = t * terms_per_thread;
        int64_t end = (t == NUM_THREADS - 1) ? NUM_TERMS : start + terms_per_thread;
        threads.emplace_back(compute_chunk, t, start, end, std::ref(results[t]));
    }

    for (auto& th : threads) th.join();

    double pi = 0.0;
    for (double r : results) pi += r;
    return pi;
}

*/