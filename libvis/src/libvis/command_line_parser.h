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

#include <string>
#include <vector>

#include "libvis/libvis.h"

namespace vis {

class CommandLineParser {
 public:
  // Constructor, does nothing. The values are not copied and therefore must
  // remain valid over the lifetime of this object.
  CommandLineParser(int argc, char** argv);
  
  // Reads a flag. Returns true if it was present and thus the value
  // was assigned. This variant for bool parameters only tests for the presence
  // of the parameter name. No value after the parameter is parsed (for example,
  // "--enable_feature false").
  bool Flag(const char* name, const char* help = "");
  
  // Reads a named parameter. Returns true if it was present and thus the value
  // was assigned. If not required, the value parameter must contain a default value.
  bool NamedParameter(const char* name, string* value, bool required = false, const char* help = "");
  
  // Reads a named parameter. Returns true if it was present and thus the value
  // was assigned. If not required, the value parameter must contain a default value.
  bool NamedParameter(const char* name, int* value, bool required = false, const char* help = "");
  
  // Reads a named parameter. Returns true if it was present and thus the value
  // was assigned. If not required, the value parameter must contain a default value.
  bool NamedParameter(const char* name, float* value, bool required = false, const char* help = "");
  
  // Reads a sequential parameter. Returns true if it was present and thus the
  // value was assigned. Sequential parameters must be read after all named
  // parameters. If not required, the value parameter must contain a default value.
  bool SequentialParameter(string* value, const char* name = "", bool required = false, const char* help = "");
  
  // Returns whether at least one of -h or --help is given in the input.
  bool HelpRequested() const;
  
  // Returns whether all required parameters were present in the input.
  bool IsInputComplete() const;
  
  // Returns whether unused parameters were present in the input (possibly
  // indicating typos).
  bool UnusedParametersGiven() const;
  
  // Combines calling HelpRequested(), IsInputComplete(), and UnusedParametersGiven().
  // Returns false if the help was requested, the input is incomplete, or there
  // was an unused parameter. Also prints out the help if requested, prints the
  // missing parameters, or prints the unused parameters.
  bool CheckParameters() const;
  
  // Prints the program usage based on all parameters specified.
  void ShowUsage() const;
  
 private:
  struct Parameter {
    string name;
    bool is_flag;
    bool is_sequential;
    bool required;
    bool given;
    string default_value;
    string help;
  };
  
  vector<Parameter> parameters_;
  
  bool is_input_complete_;
  bool help_requested_;
  
  vector<bool> value_used_;
  
  int argc_;
  char** argv_;
};

}

