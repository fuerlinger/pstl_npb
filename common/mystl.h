#ifndef MYSTL_H_INCLUDED
#define MYSTL_H_INCLUDED

#include <string>
#include <iostream>
#include <iomanip>
#include <unordered_map>

#include "util.h"


// 
// basic support for systems without PSTL: (when HAVE_PSTL is not set)
// - provide our own definitions for execution policies in the
//   std::execution namespace
//- implement the mystl algorithms in terms
//  of sequential C++11 algorithms
//

#ifdef __INTEL_COMPILER
#define HAVE_PSTL
#endif

#ifdef __INTEL_COMPILER
#include <pstl/algorithm>
#include <pstl/execution>
#include <pstl/numeric>
#else
#include <algorithm>
#include <execution>
#include <numeric>
#endif

#ifndef HAVE_PSTL
namespace std::execution
{
  class sequenced_policy {};
  class parallel_policy {};
  class parallel_unsequenced_policy {};
  
  static sequenced_policy  seq;
  static parallel_policy   par;
  static parallel_unsequenced_policy par_unseq;
}
#endif 


#define UPDATE_MAP(map_,name_, policy_,userdef_,first_,last_,iter_,time_) \
  {                                                                     \
    algo_key key;                                                       \
    key.algo    = name_;                                                \
    key.policy  = name<policy_>();                                      \
    key.userdef = userdef_;                                             \
    key.nelem   = std::distance(first_,last_);                          \
    key.elemsz  = sizeof(typename std::iterator_traits<iter_>::value_type); \
                                                                        \
    algo_val val;                                                       \
    val.count = 1;                                                      \
    val.time  = time_;                                                  \
                                                                        \
    map_[key] += val;                                                   \
  }


namespace mystd
{
  struct algo_key {
    size_t nelem;         // number of elements in the range
    size_t elemsz;        // size of each element in bytes
    std::string algo;     // name of the algorithm ("for_each", etc.)
    std::string policy;   // name of the policy ("par", "seq", etc.)
    std::string userdef;  // additional user-defined context info
        
    bool operator==(const algo_key &other) const
    {
      return (nelem   == other.nelem  &&
              elemsz  == other.elemsz &&
              algo    == other.algo   &&
              policy  == other.policy &&
              userdef == other.userdef );
    }
  };
 

  struct algo_val {
    unsigned count = 0;
    double   time  = 0.0;

    algo_val& operator+=(algo_val& rhs) {
      count += rhs.count;
      time  += rhs.time;
    }
  };
}

namespace std
{
  template <>
    struct hash<mystd::algo_key>
    {
      std::size_t operator()(const mystd::algo_key& k) const
        {
          return ((hash<size_t>()(k.nelem)
                   ^ (hash<size_t>()(k.elemsz) << 1)) >> 1);
        }
  };
}

namespace mystd
{
  std::unordered_map<algo_key, algo_val> map;
  
  template<class T> const char* name();
  template<> const char* name<decltype(std::execution::seq)&>()
    { return "seq"; }
  template<> const char* name<decltype(std::execution::par)&>()
    { return "par"; }
  template<> const char* name<decltype(std::execution::par_unseq)&>()
    { return "par_unseq"; }

  
  //
  // for_each
  //
  template<class ExecutionPolicy,
    class ForwardIt, class UnaryFunction2>
    void for_each( ExecutionPolicy&& policy,
                   ForwardIt first, ForwardIt last,
                   UnaryFunction2 f, 
		   std::string context="" )
  {
    double tbeg, tend;
    TIMESTAMP(tbeg);

#ifdef HAVE_PSTL
    std::for_each(policy,first,last,f);
#else
    std::for_each(first,last,f);
#endif
    
    TIMESTAMP(tend);
   
    UPDATE_MAP(map, "for_each", ExecutionPolicy, context,
               first, last, ForwardIt, (tend-tbeg));
  }
  
  // 
  // transform_reduce (1)
  //
  template<class ExecutionPolicy,
    class ForwardIt1, class ForwardIt2, class T>
    T transform_reduce(ExecutionPolicy&& policy,
		       ForwardIt1 first1, ForwardIt1 last1,
                       ForwardIt2 first2, T init, 
		       std::string context="")
  {
    double tbeg, tend;
    TIMESTAMP(tbeg);
#ifdef HAVE_PSTL
    T ret = std::transform_reduce(policy, first1, last1, first2, init);
#else
    T ret; // TODO
#endif
    TIMESTAMP(tend);
    
    UPDATE_MAP(map, "transform_reduce", ExecutionPolicy, context,
               first1, last1, ForwardIt1, (tend-tbeg));
    
    return ret;
  }

  //
  // transform_reduce(2)
  //
template<class ExecutionPolicy,
    class ForwardIt1, class ForwardIt2,
    class T, class BinaryOp1, class BinaryOp2>
    T transform_reduce(ExecutionPolicy&& policy,
                       ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                       T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2,
                       std::string context="")
{
  double tbeg, tend;
  TIMESTAMP(tbeg);
#ifdef HAVE_PSTL
  T ret = std::transform_reduce(policy, first1, last1, first2, init, binary_op1, binary_op2);
#else
  T ret; // TODO
#endif
  TIMESTAMP(tend);
  
  UPDATE_MAP(map, "transform_reduce", ExecutionPolicy, context,
             first1, last1, ForwardIt1, (tend-tbeg));

  return ret;
}


//
// transform_reduce (3)

template<class ExecutionPolicy,
  class ForwardIt, class T, class BinaryOp, class UnaryOp>
  T transform_reduce(ExecutionPolicy&& policy,
		     ForwardIt first, ForwardIt last,
		     T init, BinaryOp binary_op, UnaryOp unary_op,
                     std::string context="")
  {
  double tbeg, tend;
    TIMESTAMP(tbeg);
#ifdef HAVE_PSTL
    T ret = std::transform_reduce(policy, first, last, init, binary_op, unary_op);
#else
    T ret; // TODO
#endif
    TIMESTAMP(tend);

    UPDATE_MAP(map, "transform_reduce", ExecutionPolicy, context,
               first, last, ForwardIt, (tend-tbeg));
    
    return ret;
  }

    //
    // transform (1)
    //
    template< class ExecutionPolicy, 
      class ForwardIt1, class ForwardIt2, 
      class UnaryOperation >
      ForwardIt2 transform( ExecutionPolicy&& policy, 
			    ForwardIt1 first1, ForwardIt1 last1,
			    ForwardIt2 d_first, UnaryOperation unary_op,
                            std::string context="")
      {
  double tbeg, tend;
  TIMESTAMP(tbeg);

#ifdef HAVE_PSTL
  ForwardIt2 ret = std::transform(policy, first1, last1, d_first, unary_op);
#else
  ForwardIt2 ret = std::transform(first1, last1, d_first, unary_op);
#endif

  TIMESTAMP(tend);

  UPDATE_MAP(map, "transform", ExecutionPolicy, context,
	  first1, last1, ForwardIt1, (tend-tbeg));

  return ret;
}

  // 
  // transform (2)
  //
  template< class ExecutionPolicy, 
    class ForwardIt1, class ForwardIt2, class ForwardIt3, 
    class BinaryOperation >
    ForwardIt3 transform( ExecutionPolicy&& policy, ForwardIt1 first1,
                          ForwardIt1 last1, ForwardIt2 first2,
                          ForwardIt3 d_first, BinaryOperation binary_op,
                          std::string context = "")
    {
      double tbeg, tend;
      TIMESTAMP(tbeg);

#ifdef HAVE_PSTL
      ForwardIt3 ret = std::transform(policy, first1, last1, first2, d_first, binary_op);
#else
      ForwardIt3 ret = std::transform(first1, last1, first2, d_first, binary_op);
#endif
      
      TIMESTAMP(tend);

      UPDATE_MAP(map, "transform", ExecutionPolicy, context,
                 first1, last1, ForwardIt1, (tend-tbeg));
      
      return ret;
    }

  //
  // copy
  //
  template< class ExecutionPolicy,
    class ForwardIt1, class ForwardIt2 >
    ForwardIt2 copy( ExecutionPolicy&& policy,
                     ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first,
                     std::string context = "")
  {
    double tbeg, tend;
    TIMESTAMP(tbeg);

#ifdef HAVE_PSTL
    ForwardIt2 ret = std::copy(policy, first, last, d_first);
#else
    ForwardIt2 ret = std::copy(first, last, d_first);
#endif
    
    TIMESTAMP(tend);

    UPDATE_MAP(map, "copy", ExecutionPolicy, context,
               first, last, ForwardIt1, (tend-tbeg));

    return ret;
  }

  //
  // fill
  //
  template< class ExecutionPolicy, class ForwardIt, class T >
    void fill( ExecutionPolicy&& policy,
               ForwardIt first, ForwardIt last, const T& value,
               std::string context="")
  {
    double tbeg, tend;
    TIMESTAMP(tbeg);

#ifdef HAVE_PSTL
    std::fill(policy, first, last, value);
#else
    std::fill(first, last, value);
#endif
    
    TIMESTAMP(tend);

    UPDATE_MAP(map, "fill", ExecutionPolicy, context,
               first, last, ForwardIt, (tend-tbeg));

  }

  //
  // is_sorted (1)
  //
  template< class ExecutionPolicy, class ForwardIt >
    bool is_sorted( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last,
                    std::string context ="")
  {
    double tbeg, tend;
    TIMESTAMP(tbeg);
    
#ifdef HAVE_PSTL
    bool ret = std::is_sorted(policy, first, last);
#else
    bool ret = std::is_sorted(first, last);
#endif
    
    TIMESTAMP(tend);

    UPDATE_MAP(map, "is_sorted", ExecutionPolicy, context,
               first, last, ForwardIt, (tend-tbeg));
    
    return ret;
  }


  //
  // is_sorted (2)
  //
  template< class ExecutionPolicy, class ForwardIt, class Compare >
    bool is_sorted( ExecutionPolicy&& policy,
                    ForwardIt first, ForwardIt last, Compare comp,
                    std::string context="")
  {
    double tbeg, tend;
    TIMESTAMP(tbeg);

#ifdef HAVE_PSTL
    bool ret = std::is_sorted(policy, first, last, comp);
#else
    bool ret = std::is_sorted(first, last, comp);
#endif
    
    TIMESTAMP(tend);

    UPDATE_MAP(map, "is_sorted", ExecutionPolicy, context,
               first, last, ForwardIt, (tend-tbeg));
    
    return ret;
  }
  
  void dump()
  {
    using std::cout;
    using std::endl;
    using std::setw;
    
    for( auto it=map.begin(); it!=map.end(); ++it ) {
      const algo_key &key = it->first;
      const algo_val &val = it->second;

      cout << setw(20) << key.algo << " " 
	   << setw(12) << key.policy << " "
           << setw(8)  << key.nelem << " "
	   << setw(8)  << key.elemsz << " ";
      cout << setw(8)  << val.count << " "
	   << setw(14) << val.time << " "
           << setw(12) << key.userdef << endl;
    }
  }

  void clear()
  {
    map.clear();
  }
}


#endif /* MYSTL_H_INCLUDED */
