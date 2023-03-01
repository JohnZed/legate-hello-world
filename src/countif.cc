#include "legate_library.h"
#include "hello_world.h"

namespace hello {

class CountIfZeroTask : public Task<CountIfZeroTask, COUNT_IF_ZERO> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input      = context.inputs()[0];
    auto input_shape = input.shape<1>();  // should be a 1-Dim array
    auto in          = input.read_accessor<float, 1>();

    auto n      = input_shape.volume();
    float counted = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (in[i] == 0.0) counted++;
     }

    using Reduce = Legion::SumReduction<float>;
    auto countif  = context.reductions()[0].reduce_accessor<Reduce, true, 1>();
    countif.reduce(0, counted);
  }
};


class CountIfTask : public Task<CountIfTask, COUNT_IF> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input_array = context.inputs()[0];
    auto input_shape = input_array.shape<1>();  // should be a 1-Dim array
    auto in_access = input_array.read_accessor<float, 1>();

    float search_value = context.scalars()[0].value<float>();

    auto n      = input_shape.volume();
    int counted = 0;
    for (size_t i = 0; i < n; ++i) {
        if (in_access[i] == search_value) counted += 1;
     }

    using Reduce = Legion::SumReduction<int>;
    auto countif  = context.reductions()[0].reduce_accessor<Reduce, true, 1>();
    countif.reduce(0, counted);
  }
};


}  // namespace hello

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::CountIfZeroTask::register_variants();
  hello::CountIfTask::register_variants();
}

}  // namespace
