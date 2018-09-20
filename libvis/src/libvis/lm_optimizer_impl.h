// Copyright 2018 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#pragma once

namespace vis {

// HACK to determine whether class T has the variable_count() and / or
// rows() functions. This is used to allow Eigen vector types as State in
// LMOptimizer and getting the variable count using their rows() function in
// this case, while using a function with the better-suited name
// variable_count() in other cases. If neither exists, the compile error will
// complain about variable_count() missing, not rows(). Sources:
// http://stackoverflow.com/questions/12015195/how-to-call-member-function-only-if-object-happens-to-have-it
// http://stackoverflow.com/questions/29772601/why-is-sfinae-causing-failure-when-there-are-two-functions-with-different-signat
template<typename T>
struct VariableCountGetter {
  // NOTE: No function bodies are needed as they are never called.

  // If the member function A_CLASS::variable_count exists that has a compatible
  // signature, then the return type is true_type otherwise this function
  // can't exist because the type cannot be deduced.
  template <typename A_CLASS>
  static auto
      variable_count_exists(decltype(std::declval<A_CLASS>().variable_count())*)          
      -> std::true_type;

  // Member function either doesn't exist or doesn't match against the
  // required compatible signature
  template<typename A_CLASS>
  static auto
      variable_count_exists(...)
      -> std::false_type;

  // This will be of type std::true_type or std::false_type depending on the
  // result.
  typedef decltype(variable_count_exists<T>(nullptr))
      variable_count_exists_result_type;
  // This will have the value true or false depending on the result.
  static int const variable_count_exists_result =
      variable_count_exists_result_type::value;
  
  // If the member function A_CLASS::rows exists that has a compatible
  // signature, then the return type is true_type otherwise this function
  // can't exist because the type cannot be deduced.
  template <typename A_CLASS>
  static auto
      rows_exists(decltype(std::declval<A_CLASS>().rows())*)          
      -> std::true_type;

  // Member function either doesn't exist or doesn't match against the
  // required compatible signature
  template<typename A_CLASS>
  static auto
      rows_exists(...)
      -> std::false_type;

  // This will be of type std::true_type or std::false_type depending on the
  // result.
  typedef decltype(rows_exists<T>(nullptr)) rows_exists_result_type;
  // This will have the value true or false depending on the result.
  static int const rows_exists_result = rows_exists_result_type::value;
  
  // This is called if both rows() and variable_count() exist.
  static int _eval(const T& object, std::true_type, std::true_type) {
    return object.variable_count();
  }
  
  // This is called if only variable_count() exists.
  static int _eval(const T& object, std::true_type, std::false_type) {
    return object.variable_count();
  }
  
  // This is called if only rows() exists.
  static int _eval(const T& object, std::false_type, std::true_type) {
    return object.rows();
  }
  
  // This is called for otherwise unmatched arguments: neither rows() nor
  // variable_count() exist.
  static int _eval(const T& object, ...){
    // Will raise a compile error about object.variable_count() missing.
    return object.variable_count();
  }

  // Delegates to the function whose parameter types fit the types of
  // the results.
  static int eval(const T& object) {
    return _eval(object, variable_count_exists_result_type(),
                 rows_exists_result_type());
  }
};

}
