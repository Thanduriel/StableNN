#pragma once

template<typename T, typename ... Args>
struct is_callable_impl {
    template<typename C>
    static constexpr auto test(int)
        -> decltype(std::declval<C>()(std::declval<Args>() ...), bool{}) {
        return true;
    }
    template<typename> static constexpr auto test(...) { return false; }

    static constexpr bool value = test<T>(int{});
    using type = std::integral_constant<bool, value>;
};

template<typename T, typename ... Args>
using is_callable = typename is_callable_impl<T, Args...>::type;
