// Reference benchmark driver for original C++ Ninja + OneLOop.
//
// Reads one "ninjars-bench-case-v1" JSON file and, for every lane,
// evaluates ninja::Amplitude<ninja::ComplexMasses> with a
// ninja::TensorNumerator.  It writes a JSON report with
//   * correctness: Laurent coefficients, cut-constructible and rational parts;
//   * call inventories of numerator callbacks and master integrals, and the
//     exact master-integral calls (arguments and OneLOop results);
//   * pinned/repeated timings per phase-space point of
//       full            TensorNumerator + AvHOneLoop (OneLOop, no cache)
//       pure            ReplayNumerator + ReplayIntegralLibrary
//       tensor_red      TensorNumerator + ReplayIntegralLibrary
//       masters         exact recorded AvHOneLoop calls, no reduction
//       numerator       exact recorded TensorNumerator calls, no reduction
//     and the consistency check full ~= pure + (tensor_red - pure) + masters.
//
// Replays return, in call order, the outputs recorded from TensorNumerator
// and AvHOneLoop in a recording pass (O(1) copy per call).  Replayed results
// are checked to be bitwise identical to the recorded evaluation.
//
//   ninja_bench CASE.json OUT.json [--label NAME] [--block-ms 25]
//               [--rounds 9] [--no-timing]
//
// Pin the process externally (taskset -c CPU).

#include <ninja/ninja.hh>
#include <ninja/tensor_ninja.hh>
#include <ninja/avholo.hh>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ninja::Complex;
using ninja::ComplexMomentum;
using ninja::Real;
using ninja::RealMomentum;

// ---------------------------------------------------------------------------
// Minimal JSON reader (objects, arrays, numbers, strings, true/false/null).

struct Json {
  enum Type { NUL, BOOL, NUMBER, STRING, ARRAY, OBJECT } type = NUL;
  bool b = false;
  double number = 0.;
  std::string str;
  std::vector<Json> items;
  std::vector<std::pair<std::string, Json> > fields;

  const Json & operator[](const std::string & key) const
  {
    for (size_t i = 0; i < fields.size(); ++i)
      if (fields[i].first == key)
        return fields[i].second;
    throw std::runtime_error("missing JSON key: " + key);
  }
  bool has(const std::string & key) const
  {
    for (size_t i = 0; i < fields.size(); ++i)
      if (fields[i].first == key)
        return true;
    return false;
  }
  const Json & operator[](size_t i) const { return items.at(i); }
  size_t size() const { return items.size(); }
};

class JsonParser {
public:
  explicit JsonParser(const std::string & text) : s(text), i(0) {}
  Json parse()
  {
    Json v = value();
    ws();
    if (i != s.size())
      fail("trailing characters");
    return v;
  }

private:
  const std::string & s;
  size_t i;

  void fail(const char * what)
  {
    std::ostringstream o;
    o << "JSON parse error at offset " << i << ": " << what;
    throw std::runtime_error(o.str());
  }
  void ws()
  {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
      ++i;
  }
  bool lit(const char * w)
  {
    size_t n = std::strlen(w);
    if (s.compare(i, n, w) == 0) {
      i += n;
      return true;
    }
    return false;
  }
  std::string string_()
  {
    if (s[i] != '"')
      fail("expected string");
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\') {
        ++i;
        char c = s[i];
        if (c == 'n') out += '\n';
        else if (c == 't') out += '\t';
        else if (c == 'u') { out += '?'; i += 4; }
        else out += c;
        ++i;
      } else {
        out += s[i++];
      }
    }
    if (i >= s.size())
      fail("unterminated string");
    ++i;
    return out;
  }
  Json value()
  {
    ws();
    if (i >= s.size())
      fail("unexpected end");
    Json v;
    char c = s[i];
    if (c == '{') {
      v.type = Json::OBJECT;
      ++i;
      ws();
      if (s[i] == '}') { ++i; return v; }
      for (;;) {
        ws();
        std::string key = string_();
        ws();
        if (s[i] != ':') fail("expected ':'");
        ++i;
        v.fields.push_back(std::make_pair(key, value()));
        ws();
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == '}') { ++i; break; }
        fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      v.type = Json::ARRAY;
      ++i;
      ws();
      if (s[i] == ']') { ++i; return v; }
      for (;;) {
        v.items.push_back(value());
        ws();
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == ']') { ++i; break; }
        fail("expected ',' or ']'");
      }
    } else if (c == '"') {
      v.type = Json::STRING;
      v.str = string_();
    } else if (lit("true")) {
      v.type = Json::BOOL; v.b = true;
    } else if (lit("false")) {
      v.type = Json::BOOL; v.b = false;
    } else if (lit("null")) {
      v.type = Json::NUL;
    } else {
      const char * start = s.c_str() + i;
      char * end = 0;
      v.type = Json::NUMBER;
      v.number = std::strtod(start, &end);
      if (end == start)
        fail("bad value");
      i += end - start;
    }
    return v;
  }
};

// ---------------------------------------------------------------------------
// Case data.

struct Lane {
  std::string id;
  std::vector<RealMomentum> V;
  std::vector<Complex> m2;
  Real mu2;
  bool has_s;
  std::vector<Real> s;
  std::vector<Complex> tensor;
};

struct Case {
  std::string name;
  int n, rank;
  std::vector<Lane> lanes;
};

static int ncoeff(int r) { return (r + 1) * (r + 2) * (r + 3) * (r + 4) / 24; }

static Case read_case(const std::string & path)
{
  std::ifstream in(path.c_str());
  if (!in)
    throw std::runtime_error("cannot open " + path);
  std::stringstream buf;
  buf << in.rdbuf();
  std::string text = buf.str();
  Json j = JsonParser(text).parse();
  if (j["format"].str != "ninjars-bench-case-v1")
    throw std::runtime_error("unsupported case format " + j["format"].str);
  Case c;
  c.name = j["name"].str;
  c.n = int(j["n"].number);
  c.rank = int(j["rank"].number);
  const Json & lanes = j["lanes"];
  for (size_t l = 0; l < lanes.size(); ++l) {
    const Json & lj = lanes[l];
    Lane lane;
    lane.id = lj["id"].str;
    const Json & shifts = lj["shifts"];
    const Json & masses = lj["masses_squared"];
    if (int(shifts.size()) != c.n || int(masses.size()) != c.n)
      throw std::runtime_error("lane " + lane.id + ": wrong number of shifts/masses");
    for (int i = 0; i < c.n; ++i) {
      lane.V.push_back(RealMomentum(shifts[i][0].number, shifts[i][1].number,
                                    shifts[i][2].number, shifts[i][3].number));
      lane.m2.push_back(Complex(masses[i][0].number, masses[i][1].number));
    }
    lane.mu2 = lj["mu_squared"].number;
    lane.has_s = lj.has("s_matrix") && lj["s_matrix"].type == Json::ARRAY;
    if (lane.has_s) {
      const Json & sm = lj["s_matrix"];
      lane.s.resize(c.n * c.n);
      for (int i = 0; i < c.n; ++i)
        for (int k = 0; k < c.n; ++k)
          lane.s[i * c.n + k] = sm[i][k].number;
    }
    const Json & tensor = lj["tensor"];
    if (int(tensor.size()) != ncoeff(c.rank))
      throw std::runtime_error("lane " + lane.id + ": tensor length mismatch");
    for (size_t k = 0; k < tensor.size(); ++k)
      lane.tensor.push_back(Complex(tensor[k][0].number, tensor[k][1].number));
    c.lanes.push_back(lane);
  }
  return c;
}

// ---------------------------------------------------------------------------
// Numerator recording and replay.

enum NumKind { NK_EVALUATE = 0, NK_MU = 1, NK_T3 = 2, NK_T2 = 3, NK_COUNT = 4 };
static const char * num_kind_name[NK_COUNT] = {"evaluate", "muExpansion", "t3Expansion", "t2Expansion"};

struct NumCall {
  int kind;
  int mindeg, cut, nout;
  ComplexMomentum v[4];
  Complex param[3];
};

struct NumTape {
  std::vector<NumCall> calls;
  std::vector<Complex> outputs;   // concatenated outputs in call order
};

static const int SCRATCH = 128;

static void fill_sentinel(Complex * c)
{
  const double nan = std::nan("0x5a5a");
  for (int i = 0; i < SCRATCH; ++i)
    c[i] = Complex(nan, nan);
}

static bool is_sentinel(const Complex & c)
{
  const double nan = std::nan("0x5a5a");
  double re = c.real(), im = c.imag();
  return std::memcmp(&re, &nan, sizeof(double)) == 0 && std::memcmp(&im, &nan, sizeof(double)) == 0;
}

// Count written prefix and check that nothing beyond it was written.
static int written(const Complex * c)
{
  int last = -1;
  for (int i = 0; i < SCRATCH; ++i)
    if (!is_sentinel(c[i]))
      last = i;
  for (int i = 0; i <= last; ++i)
    if (is_sentinel(c[i]))
      throw std::runtime_error("numerator output is not a contiguous prefix");
  return last + 1;
}

class RecordingNumerator : public ninja::Numerator {
public:
  RecordingNumerator(ninja::Numerator & inner, NumTape & tape, int n, int rank)
    : inner_(inner), tape_(tape), n_(n), rank_(rank) {}

  virtual Complex evaluate(const ComplexMomentum & q, const Complex & muq,
                           int cut, const ninja::PartitionInt part[])
  {
    NumCall call;
    call.kind = NK_EVALUATE; call.mindeg = 0; call.cut = cut; call.nout = 1;
    call.v[0] = q; call.param[0] = muq;
    Complex r = inner_.evaluate(q, muq, cut, part);
    tape_.calls.push_back(call);
    tape_.outputs.push_back(r);
    return r;
  }

  virtual void muExpansion(const ComplexMomentum v_perp[], const ninja::PartitionInt part[],
                           Complex c[])
  {
    NumCall call;
    call.kind = NK_MU; call.mindeg = 0; call.cut = 0;
    call.v[0] = v_perp[0];
    if (rank_ >= n_ + 1)
      call.v[1] = v_perp[1];
    Complex scratch[SCRATCH];
    fill_sentinel(scratch);
    inner_.muExpansion(v_perp, part, scratch);
    finish(call, scratch, c);
  }

  virtual void t3Expansion(const ComplexMomentum & a, const ComplexMomentum & e3,
                           const ComplexMomentum & e4, const Complex & param, int mindeg,
                           int cut, const ninja::PartitionInt part[], Complex c[])
  {
    NumCall call;
    call.kind = NK_T3; call.mindeg = mindeg; call.cut = cut;
    call.v[0] = a; call.v[1] = e3; call.v[2] = e4; call.param[0] = param;
    Complex scratch[SCRATCH];
    fill_sentinel(scratch);
    inner_.t3Expansion(a, e3, e4, param, mindeg, cut, part, scratch);
    finish(call, scratch, c);
  }

  virtual void t2Expansion(const ComplexMomentum & a0, const ComplexMomentum & a1,
                           const ComplexMomentum & e3, const ComplexMomentum & e4,
                           const Complex param[], int mindeg, int cut,
                           const ninja::PartitionInt part[], Complex c[])
  {
    NumCall call;
    call.kind = NK_T2; call.mindeg = mindeg; call.cut = cut;
    call.v[0] = a0; call.v[1] = a1; call.v[2] = e3; call.v[3] = e4;
    call.param[0] = param[0]; call.param[1] = param[1]; call.param[2] = param[2];
    Complex scratch[SCRATCH];
    fill_sentinel(scratch);
    inner_.t2Expansion(a0, a1, e3, e4, param, mindeg, cut, part, scratch);
    finish(call, scratch, c);
  }

private:
  void finish(NumCall & call, const Complex * scratch, Complex * c)
  {
    call.nout = written(scratch);
    for (int i = 0; i < call.nout; ++i) {
      c[i] = scratch[i];
      tape_.outputs.push_back(scratch[i]);
    }
    tape_.calls.push_back(call);
  }

  ninja::Numerator & inner_;
  NumTape & tape_;
  int n_, rank_;
};

class ReplayNumerator : public ninja::Numerator {
public:
  ReplayNumerator() : out_(0), counts_(0), pos_(0), call_(0) {}
  void bind(const NumTape & tape)
  {
    counts_store_.clear();
    for (size_t i = 0; i < tape.calls.size(); ++i)
      counts_store_.push_back(tape.calls[i].nout);
    outputs_store_ = tape.outputs;
    out_ = outputs_store_.data();
    counts_ = counts_store_.data();
    ncalls_ = counts_store_.size();
    rewind();
  }
  void rewind() { pos_ = 0; call_ = 0; }
  bool consumed() const { return call_ == ncalls_ && pos_ == outputs_store_.size(); }

  virtual Complex evaluate(const ComplexMomentum &, const Complex &, int, const ninja::PartitionInt[])
  {
    ++call_;
    return out_[pos_++];
  }
  virtual void muExpansion(const ComplexMomentum[], const ninja::PartitionInt[], Complex c[])
  {
    copy(c);
  }
  virtual void t3Expansion(const ComplexMomentum &, const ComplexMomentum &, const ComplexMomentum &,
                           const Complex &, int, int, const ninja::PartitionInt[], Complex c[])
  {
    copy(c);
  }
  virtual void t2Expansion(const ComplexMomentum &, const ComplexMomentum &, const ComplexMomentum &,
                           const ComplexMomentum &, const Complex[], int, int,
                           const ninja::PartitionInt[], Complex c[])
  {
    copy(c);
  }

private:
  inline void copy(Complex * c)
  {
    int k = counts_[call_++];
    std::memcpy(static_cast<void *>(c), out_ + pos_, k * sizeof(Complex));
    pos_ += k;
  }
  std::vector<Complex> outputs_store_;
  std::vector<int> counts_store_;
  const Complex * out_;
  const int * counts_;
  size_t ncalls_;
  size_t pos_, call_;
};

// Re-issue the recorded numerator calls (inputs) without reduction.
static Complex replay_numerator_inputs(ninja::Numerator & num, const NumTape & tape)
{
  static const ninja::PartitionInt part[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  Complex sink = 0.;
  Complex c[SCRATCH];
  for (size_t i = 0; i < tape.calls.size(); ++i) {
    const NumCall & k = tape.calls[i];
    switch (k.kind) {
    case NK_EVALUATE:
      sink += num.evaluate(k.v[0], k.param[0], k.cut, part);
      break;
    case NK_MU:
      num.muExpansion(k.v, part, c);
      sink += c[0];
      break;
    case NK_T3:
      num.t3Expansion(k.v[0], k.v[1], k.v[2], k.param[0], k.mindeg, k.cut, part, c);
      sink += c[0];
      break;
    case NK_T2:
      num.t2Expansion(k.v[0], k.v[1], k.v[2], k.v[3], k.param, k.mindeg, k.cut, part, c);
      sink += c[0];
      break;
    }
  }
  return sink;
}

// ---------------------------------------------------------------------------
// Master-integral recording and replay.

enum MasterKind {
  MK_BOX_CM, MK_BOX_RM, MK_BOX_NM, MK_TRI_CM, MK_TRI_RM, MK_TRI_NM,
  MK_BUB_CM, MK_BUB_RM, MK_BUB_NM, MK_B2_CM, MK_B2_RM, MK_B2_NM,
  MK_B3_CM, MK_B3_RM, MK_B3_NM, MK_TAD_CM, MK_TAD_RM, MK_TAD_NM, MK_COUNT
};
static const char * master_kind_name[MK_COUNT] = {
  "box_cm", "box_rm", "box_nm", "triangle_cm", "triangle_rm", "triangle_nm",
  "bubble_cm", "bubble_rm", "bubble_nm", "rank2_bubble_cm", "rank2_bubble_rm", "rank2_bubble_nm",
  "rank3_bubble_cm", "rank3_bubble_rm", "rank3_bubble_nm", "tadpole_cm", "tadpole_rm", "tadpole_nm"};
static int master_nout(int kind)
{
  if (kind >= MK_B2_CM && kind <= MK_B2_NM) return 9;
  if (kind >= MK_B3_CM && kind <= MK_B3_NM) return 12;
  return 3;
}

enum MasterGroup { MG_BOX, MG_TRIANGLE, MG_BUBBLE, MG_TADPOLE, MG_COUNT };
static const char * master_group_name[MG_COUNT] = {"boxes", "triangles", "bubbles", "tadpoles"};
static int master_group(int kind)
{
  if (kind <= MK_BOX_NM) return MG_BOX;
  if (kind <= MK_TRI_NM) return MG_TRIANGLE;
  if (kind <= MK_B3_NM) return MG_BUBBLE;
  return MG_TADPOLE;
}

struct MasterCall {
  int kind;
  Real s[6];
  Complex m[4];
  Real mu2;
};

struct MasterTape {
  std::vector<MasterCall> calls;
  std::vector<Complex> outputs;
  std::vector<Real> inits;
};

// Dispatch one call to an IntegralLibrary; out has master_nout(kind) slots.
static void issue(ninja::IntegralLibrary & lib, const MasterCall & k, Complex * out)
{
  const Real * s = k.s;
  const Complex * m = k.m;
  switch (k.kind) {
  case MK_BOX_CM: lib.getBoxIntegralCM(out, s[0], s[1], s[2], s[3], s[4], s[5], m[0], m[1], m[2], m[3]); break;
  case MK_BOX_RM: lib.getBoxIntegralRM(out, s[0], s[1], s[2], s[3], s[4], s[5], m[0].real(), m[1].real(), m[2].real(), m[3].real()); break;
  case MK_BOX_NM: lib.getBoxIntegralNM(out, s[0], s[1], s[2], s[3], s[4], s[5]); break;
  case MK_TRI_CM: lib.getTriangleIntegralCM(out, s[0], s[1], s[2], m[0], m[1], m[2]); break;
  case MK_TRI_RM: lib.getTriangleIntegralRM(out, s[0], s[1], s[2], m[0].real(), m[1].real(), m[2].real()); break;
  case MK_TRI_NM: lib.getTriangleIntegralNM(out, s[0], s[1], s[2]); break;
  case MK_BUB_CM: lib.getBubbleIntegralCM(out, s[0], m[0], m[1]); break;
  case MK_BUB_RM: lib.getBubbleIntegralRM(out, s[0], m[0].real(), m[1].real()); break;
  case MK_BUB_NM: lib.getBubbleIntegralNM(out, s[0]); break;
  case MK_B2_CM: lib.getRank2BubbleIntegralCM(out, out + 3, out + 6, s[0], m[0], m[1]); break;
  case MK_B2_RM: lib.getRank2BubbleIntegralRM(out, out + 3, out + 6, s[0], m[0].real(), m[1].real()); break;
  case MK_B2_NM: lib.getRank2BubbleIntegralNM(out, out + 3, out + 6, s[0]); break;
  case MK_B3_CM: lib.getRank3BubbleIntegralCM(out, out + 3, out + 6, out + 9, s[0], m[0], m[1]); break;
  case MK_B3_RM: lib.getRank3BubbleIntegralRM(out, out + 3, out + 6, out + 9, s[0], m[0].real(), m[1].real()); break;
  case MK_B3_NM: lib.getRank3BubbleIntegralNM(out, out + 3, out + 6, out + 9, s[0]); break;
  case MK_TAD_CM: lib.getTadpoleIntegralCM(out, m[0]); break;
  case MK_TAD_RM: lib.getTadpoleIntegralRM(out, m[0].real()); break;
  case MK_TAD_NM: lib.getTadpoleIntegralNM(out); break;
  }
}

class RecordingLibrary : public ninja::IntegralLibrary {
public:
  RecordingLibrary(ninja::IntegralLibrary & inner, MasterTape & tape) : inner_(inner), tape_(tape), mu2_(0) {}

  virtual void init(Real muRsq) { mu2_ = muRsq; tape_.inits.push_back(muRsq); inner_.init(muRsq); }
  virtual void exit() { inner_.exit(); }

#define REC_BEGIN(KIND) MasterCall k; k.kind = KIND; k.mu2 = mu2_; \
  { for (int i_ = 0; i_ < 6; ++i_) k.s[i_] = 0.; } { for (int i_ = 0; i_ < 4; ++i_) k.m[i_] = 0.; }
#define REC_END(OUT) { issue(inner_, k, OUT); tape_.calls.push_back(k); \
  for (int i_ = 0; i_ < master_nout(k.kind); ++i_) tape_.outputs.push_back(OUT[i_]); }

  virtual void getBoxIntegralRM(Complex r[3], Real s21, Real s32, Real s43, Real s14, Real s31, Real s42,
                                Real m1, Real m2, Real m3, Real m4)
  { REC_BEGIN(MK_BOX_RM) set6(k, s21, s32, s43, s14, s31, s42); k.m[0]=m1; k.m[1]=m2; k.m[2]=m3; k.m[3]=m4; REC_END(r) }
  virtual void getBoxIntegralCM(Complex r[3], Real s21, Real s32, Real s43, Real s14, Real s31, Real s42,
                                const Complex & m1, const Complex & m2, const Complex & m3, const Complex & m4)
  { REC_BEGIN(MK_BOX_CM) set6(k, s21, s32, s43, s14, s31, s42); k.m[0]=m1; k.m[1]=m2; k.m[2]=m3; k.m[3]=m4; REC_END(r) }
  virtual void getBoxIntegralNM(Complex r[3], Real s21, Real s32, Real s43, Real s14, Real s31, Real s42)
  { REC_BEGIN(MK_BOX_NM) set6(k, s21, s32, s43, s14, s31, s42); REC_END(r) }
  virtual void getTriangleIntegralRM(Complex r[3], Real s21, Real s32, Real s13, Real m1, Real m2, Real m3)
  { REC_BEGIN(MK_TRI_RM) k.s[0]=s21; k.s[1]=s32; k.s[2]=s13; k.m[0]=m1; k.m[1]=m2; k.m[2]=m3; REC_END(r) }
  virtual void getTriangleIntegralCM(Complex r[3], Real s21, Real s32, Real s13,
                                     const Complex & m1, const Complex & m2, const Complex & m3)
  { REC_BEGIN(MK_TRI_CM) k.s[0]=s21; k.s[1]=s32; k.s[2]=s13; k.m[0]=m1; k.m[1]=m2; k.m[2]=m3; REC_END(r) }
  virtual void getTriangleIntegralNM(Complex r[3], Real s21, Real s32, Real s13)
  { REC_BEGIN(MK_TRI_NM) k.s[0]=s21; k.s[1]=s32; k.s[2]=s13; REC_END(r) }
  virtual void getBubbleIntegralRM(Complex r[3], Real s21, Real m1, Real m2)
  { REC_BEGIN(MK_BUB_RM) k.s[0]=s21; k.m[0]=m1; k.m[1]=m2; REC_END(r) }
  virtual void getBubbleIntegralCM(Complex r[3], Real s21, const Complex & m1, const Complex & m2)
  { REC_BEGIN(MK_BUB_CM) k.s[0]=s21; k.m[0]=m1; k.m[1]=m2; REC_END(r) }
  virtual void getBubbleIntegralNM(Complex r[3], Real s21)
  { REC_BEGIN(MK_BUB_NM) k.s[0]=s21; REC_END(r) }
  virtual void getRank2BubbleIntegralRM(Complex b11[3], Complex b1[3], Complex b0[3], Real s21, Real m1, Real m2)
  { REC_BEGIN(MK_B2_RM) k.s[0]=s21; k.m[0]=m1; k.m[1]=m2; Complex o[9]; REC_END(o) unpack(o, b11, b1, b0); }
  virtual void getRank2BubbleIntegralCM(Complex b11[3], Complex b1[3], Complex b0[3], Real s21,
                                        const Complex & m1, const Complex & m2)
  { REC_BEGIN(MK_B2_CM) k.s[0]=s21; k.m[0]=m1; k.m[1]=m2; Complex o[9]; REC_END(o) unpack(o, b11, b1, b0); }
  virtual void getRank2BubbleIntegralNM(Complex b11[3], Complex b1[3], Complex b0[3], Real s21)
  { REC_BEGIN(MK_B2_NM) k.s[0]=s21; Complex o[9]; REC_END(o) unpack(o, b11, b1, b0); }
  virtual void getRank3BubbleIntegralRM(Complex b111[3], Complex b11[3], Complex b1[3], Complex b0[3],
                                        Real s21, Real m1, Real m2)
  { REC_BEGIN(MK_B3_RM) k.s[0]=s21; k.m[0]=m1; k.m[1]=m2; Complex o[12]; REC_END(o) unpack4(o, b111, b11, b1, b0); }
  virtual void getRank3BubbleIntegralCM(Complex b111[3], Complex b11[3], Complex b1[3], Complex b0[3],
                                        Real s21, const Complex & m1, const Complex & m2)
  { REC_BEGIN(MK_B3_CM) k.s[0]=s21; k.m[0]=m1; k.m[1]=m2; Complex o[12]; REC_END(o) unpack4(o, b111, b11, b1, b0); }
  virtual void getRank3BubbleIntegralNM(Complex b111[3], Complex b11[3], Complex b1[3], Complex b0[3], Real s21)
  { REC_BEGIN(MK_B3_NM) k.s[0]=s21; Complex o[12]; REC_END(o) unpack4(o, b111, b11, b1, b0); }
  virtual void getTadpoleIntegralRM(Complex r[3], Real m0)
  { REC_BEGIN(MK_TAD_RM) k.m[0]=m0; REC_END(r) }
  virtual void getTadpoleIntegralCM(Complex r[3], const Complex & m0)
  { REC_BEGIN(MK_TAD_CM) k.m[0]=m0; REC_END(r) }
  virtual void getTadpoleIntegralNM(Complex r[3])
  { REC_BEGIN(MK_TAD_NM) REC_END(r) }

private:
  static void set6(MasterCall & k, Real a, Real b, Real c, Real d, Real e, Real f)
  { k.s[0]=a; k.s[1]=b; k.s[2]=c; k.s[3]=d; k.s[4]=e; k.s[5]=f; }
  static void unpack(const Complex * o, Complex * a, Complex * b, Complex * c)
  { for (int i = 0; i < 3; ++i) { a[i] = o[i]; b[i] = o[3+i]; c[i] = o[6+i]; } }
  static void unpack4(const Complex * o, Complex * a, Complex * b, Complex * c, Complex * d)
  { for (int i = 0; i < 3; ++i) { a[i] = o[i]; b[i] = o[3+i]; c[i] = o[6+i]; d[i] = o[9+i]; } }

  ninja::IntegralLibrary & inner_;
  MasterTape & tape_;
  Real mu2_;
};

// Returns recorded master results in call order.  Overrides every method
// Ninja may call (including the rank-3 and massless defaults), so the
// replayed sequence is exactly the recorded interface sequence.
class ReplayLibrary : public ninja::IntegralLibrary {
public:
  ReplayLibrary() : out_(0), pos_(0) {}
  void bind(const MasterTape & tape) { store_ = tape.outputs; out_ = store_.data(); rewind(); }
  void rewind() { pos_ = 0; }
  bool consumed() const { return pos_ == store_.size(); }

  virtual void init(Real) {}
  virtual void getBoxIntegralRM(Complex r[3], Real, Real, Real, Real, Real, Real, Real, Real, Real, Real) { c3(r); }
  virtual void getBoxIntegralCM(Complex r[3], Real, Real, Real, Real, Real, Real,
                                const Complex &, const Complex &, const Complex &, const Complex &) { c3(r); }
  virtual void getBoxIntegralNM(Complex r[3], Real, Real, Real, Real, Real, Real) { c3(r); }
  virtual void getTriangleIntegralRM(Complex r[3], Real, Real, Real, Real, Real, Real) { c3(r); }
  virtual void getTriangleIntegralCM(Complex r[3], Real, Real, Real, const Complex &, const Complex &, const Complex &) { c3(r); }
  virtual void getTriangleIntegralNM(Complex r[3], Real, Real, Real) { c3(r); }
  virtual void getBubbleIntegralRM(Complex r[3], Real, Real, Real) { c3(r); }
  virtual void getBubbleIntegralCM(Complex r[3], Real, const Complex &, const Complex &) { c3(r); }
  virtual void getBubbleIntegralNM(Complex r[3], Real) { c3(r); }
  virtual void getRank2BubbleIntegralRM(Complex a[3], Complex b[3], Complex c[3], Real, Real, Real) { c3(a); c3(b); c3(c); }
  virtual void getRank2BubbleIntegralCM(Complex a[3], Complex b[3], Complex c[3], Real, const Complex &, const Complex &) { c3(a); c3(b); c3(c); }
  virtual void getRank2BubbleIntegralNM(Complex a[3], Complex b[3], Complex c[3], Real) { c3(a); c3(b); c3(c); }
  virtual void getRank3BubbleIntegralRM(Complex a[3], Complex b[3], Complex c[3], Complex d[3], Real, Real, Real) { c3(a); c3(b); c3(c); c3(d); }
  virtual void getRank3BubbleIntegralCM(Complex a[3], Complex b[3], Complex c[3], Complex d[3], Real, const Complex &, const Complex &) { c3(a); c3(b); c3(c); c3(d); }
  virtual void getRank3BubbleIntegralNM(Complex a[3], Complex b[3], Complex c[3], Complex d[3], Real) { c3(a); c3(b); c3(c); c3(d); }
  virtual void getTadpoleIntegralRM(Complex r[3], Real) { c3(r); }
  virtual void getTadpoleIntegralCM(Complex r[3], const Complex &) { c3(r); }
  virtual void getTadpoleIntegralNM(Complex r[3]) { c3(r); }

private:
  inline void c3(Complex * r)
  {
    r[0] = out_[pos_]; r[1] = out_[pos_ + 1]; r[2] = out_[pos_ + 2];
    pos_ += 3;
  }
  std::vector<Complex> store_;
  const Complex * out_;
  size_t pos_;
};

// Re-issue the recorded master calls through AvHOneLoop (no reduction).
static Complex replay_masters(ninja::IntegralLibrary & lib, const MasterTape & tape, Real mu2)
{
  Complex sink = 0.;
  Complex out[12];
  lib.init(mu2);
  for (size_t i = 0; i < tape.calls.size(); ++i) {
    issue(lib, tape.calls[i], out);
    sink += out[0];
  }
  lib.exit();
  return sink;
}

// ---------------------------------------------------------------------------
// Per-lane state.

struct LaneState {
  Lane * lane;
  ninja::Amplitude<ninja::ComplexMasses> * amp;
  ninja::TensorNumerator * num;
  NumTape num_tape;
  MasterTape master_tape;
  MasterTape master_groups[MG_COUNT];
  ReplayNumerator replay_num;
  ReplayLibrary replay_lib;
  // results
  int status;
  Complex eps[3], cc, rat;
};

struct Result { int status; Complex eps[3], cc, rat; };

static Result evaluate(ninja::Amplitude<ninja::ComplexMasses> & amp, ninja::Numerator & num,
                       ninja::IntegralLibrary & lib)
{
  amp.reset();
  amp.setIntegralLibrary(lib);
  Result r;
  r.status = amp.evaluate(num);
  r.eps[0] = amp[0]; r.eps[1] = amp[1]; r.eps[2] = amp[2];
  r.cc = amp.getCutConstructiblePart();
  r.rat = amp.getRationalPart();
  return r;
}

static bool same_bits(const Complex & a, const Complex & b)
{
  return std::memcmp(&a, &b, sizeof(Complex)) == 0;
}

static bool same_result(const Result & a, const Result & b)
{
  return a.status == b.status && same_bits(a.eps[0], b.eps[0]) && same_bits(a.eps[1], b.eps[1])
    && same_bits(a.eps[2], b.eps[2]) && same_bits(a.cc, b.cc) && same_bits(a.rat, b.rat);
}

// ---------------------------------------------------------------------------
// Timing.

typedef std::chrono::steady_clock Clock;
static volatile double g_sink = 0.;

struct Mode {
  std::string name;
  std::string description;
  long iterations;              // evaluations of all lanes per block
  std::vector<double> ns_per_point;
};

template <typename F>
static double run_block(F & f, int nlanes, long iterations)
{
  Clock::time_point t0 = Clock::now();
  for (long it = 0; it < iterations; ++it)
    for (int l = 0; l < nlanes; ++l)
      f(l);
  Clock::time_point t1 = Clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

template <typename F>
static long calibrate(F & f, int nlanes, double block_ns)
{
  long it = 1;
  for (;;) {
    double t = run_block(f, nlanes, it);
    if (t >= 0.2 * block_ns || it > (1L << 40)) {
      double per = t / it;
      long need = long(std::ceil(block_ns / per));
      return std::max(1L, need);
    }
    it *= 4;
  }
}


// MadLoop's C entry points (ninja_wraps.cc): construct a TensorNumerator and an
// Amplitude per call and use the default integral library (AvHOneLoop).
extern "C" {
  void ninja_tensor_evaluate_cm_(const Complex * tensor, const int & n, const int & rank,
                                 const RealMomentum p[], const Complex m2[], const Real & mur2,
                                 Complex tot[3], Complex & totr, int & return_status);
  void ninja_tensor_evaluate_smat_cm_(const Complex * tensor, const int & n, const int & rank,
                                      Real * s_mat, const RealMomentum p[], const Complex m2[],
                                      const Real & mur2, Complex tot[3], Complex & totr,
                                      int & return_status);
}

static Result entry_point(Lane & lane, int n, int rank)
{
  Result r;
  Complex tot[3], totr;
  if (lane.has_s)
    ninja_tensor_evaluate_smat_cm_(lane.tensor.data(), n, rank, lane.s.data(), lane.V.data(),
                                   lane.m2.data(), lane.mu2, tot, totr, r.status);
  else
    ninja_tensor_evaluate_cm_(lane.tensor.data(), n, rank, lane.V.data(), lane.m2.data(),
                              lane.mu2, tot, totr, r.status);
  r.eps[0] = tot[0]; r.eps[1] = tot[1]; r.eps[2] = tot[2];
  r.rat = totr;
  r.cc = tot[0] - totr;
  return r;
}

static void add_mode(std::vector<Mode> & modes, std::vector<std::function<void(int)> > & fns,
                     const std::string & name, const std::string & description,
                     std::function<void(int)> fn)
{
  Mode mode;
  mode.name = name;
  mode.description = description;
  mode.iterations = 0;
  modes.push_back(mode);
  fns.push_back(fn);
}

static double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n == 0) return std::nan("");
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------------------
// Output helpers.

static std::string jnum(double x)
{
  if (!std::isfinite(x)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}
static std::string jc(const Complex & z) { return "[" + jnum(z.real()) + "," + jnum(z.imag()) + "]"; }
static std::string jstr(const std::string & s)
{
  std::string o = "\"";
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '"' || s[i] == '\\') o += '\\';
    o += s[i];
  }
  return o + "\"";
}

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::cerr << "usage: ninja_bench CASE.json OUT.json [--label NAME] [--block-ms 25] [--rounds 9] [--no-timing]\n";
    return 2;
  }
  std::string case_path = argv[1], out_path = argv[2], label = "unlabelled";
  double block_ms = 25.;
  int rounds = 9;
  bool timing = true;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--label" && i + 1 < argc) label = argv[++i];
    else if (a == "--block-ms" && i + 1 < argc) block_ms = std::atof(argv[++i]);
    else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (a == "--no-timing") timing = false;
    else { std::cerr << "unknown argument " << a << "\n"; return 2; }
  }

  {
    std::ostream null_stream(0);
    ninja::printBanner(null_stream, true);   // also silences later banners
  }

  Case c = read_case(case_path);
  const int L = int(c.lanes.size());
  std::vector<LaneState> st(L);
  for (int l = 0; l < L; ++l) {
    Lane & lane = c.lanes[l];
    st[l].lane = &lane;
    st[l].amp = new ninja::Amplitude<ninja::ComplexMasses>(c.n, c.rank, lane.V.data(), lane.m2.data());
    st[l].amp->setRenormalizationScale(lane.mu2);
    if (lane.has_s)
      st[l].amp->setSMatrix(lane.s.data());
    st[l].num = new ninja::TensorNumerator(c.n, c.rank, lane.tensor.data());
  }

  // Reference evaluation, recording pass, and bitwise replay checks.
  std::vector<Result> reference(L);
  bool all_identical = true;
  std::vector<std::string> lane_checks(L);
  for (int l = 0; l < L; ++l) {
    LaneState & s = st[l];
    reference[l] = evaluate(*s.amp, *s.num, ninja::avh_olo);
    RecordingNumerator rec_num(*s.num, s.num_tape, c.n, c.rank);
    RecordingLibrary rec_lib(ninja::avh_olo, s.master_tape);
    Result recorded = evaluate(*s.amp, rec_num, rec_lib);
    s.replay_num.bind(s.num_tape);
    s.replay_lib.bind(s.master_tape);
    Result pure = evaluate(*s.amp, s.replay_num, s.replay_lib);
    bool pure_consumed = s.replay_num.consumed() && s.replay_lib.consumed();
    s.replay_lib.rewind();
    Result tensor_red = evaluate(*s.amp, *s.num, s.replay_lib);
    bool lib_consumed = s.replay_lib.consumed();
    // masters-only replay reproduces the recorded OneLOop outputs
    bool masters_ok = true;
    {
      Complex out[12];
      ninja::avh_olo.init(s.lane->mu2);
      size_t pos = 0;
      for (size_t i = 0; i < s.master_tape.calls.size(); ++i) {
        issue(ninja::avh_olo, s.master_tape.calls[i], out);
        for (int k = 0; k < master_nout(s.master_tape.calls[i].kind); ++k)
          masters_ok = masters_ok && same_bits(out[k], s.master_tape.outputs[pos++]);
      }
      ninja::avh_olo.exit();
    }
    {
      size_t pos2 = 0;
      for (size_t i = 0; i < s.master_tape.calls.size(); ++i) {
        const MasterCall & k = s.master_tape.calls[i];
        MasterTape & g = s.master_groups[master_group(k.kind)];
        g.calls.push_back(k);
        for (int j = 0; j < master_nout(k.kind); ++j)
          g.outputs.push_back(s.master_tape.outputs[pos2++]);
      }
    }
    Result entry = entry_point(*s.lane, c.n, c.rank);
    bool entry_ok = entry.status == reference[l].status && same_bits(entry.eps[0], reference[l].eps[0])
      && same_bits(entry.eps[1], reference[l].eps[1]) && same_bits(entry.eps[2], reference[l].eps[2])
      && same_bits(entry.rat, reference[l].rat);
    bool ok = same_result(reference[l], recorded) && same_result(reference[l], pure)
      && same_result(reference[l], tensor_red) && pure_consumed && lib_consumed && masters_ok && entry_ok;
    all_identical = all_identical && ok;
    std::ostringstream o;
    o << "{\"recorded_identical\":" << (same_result(reference[l], recorded) ? "true" : "false")
      << ",\"pure_replay_identical\":" << (same_result(reference[l], pure) ? "true" : "false")
      << ",\"tensor_reduction_identical\":" << (same_result(reference[l], tensor_red) ? "true" : "false")
      << ",\"replays_fully_consumed\":" << (pure_consumed && lib_consumed ? "true" : "false")
      << ",\"masters_replay_identical\":" << (masters_ok ? "true" : "false")
      << ",\"madloop_entry_identical\":" << (entry_ok ? "true" : "false") << "}";
    lane_checks[l] = o.str();
  }

  // Timings: interleaved rounds, one block per mode per round.
  std::vector<Mode> modes;
  if (timing) {
    const double block_ns = block_ms * 1e6;
    std::vector<std::function<void(int)> > fns;
    std::vector<LaneState> * sp = &st;
    const int nn = c.n, rr = c.rank;
    add_mode(modes, fns, "full", "TensorNumerator + AvHOneLoop (OneLOop, cache disabled), Amplitude::evaluate",
             [sp](int l) { LaneState & s = (*sp)[l]; Result r = evaluate(*s.amp, *s.num, ninja::avh_olo); g_sink += r.eps[0].real(); });
    add_mode(modes, fns, "pure_reduction", "ReplayNumerator + ReplayIntegralLibrary, Amplitude::evaluate",
             [sp](int l) { LaneState & s = (*sp)[l]; s.replay_num.rewind(); s.replay_lib.rewind();
                           Result r = evaluate(*s.amp, s.replay_num, s.replay_lib); g_sink += r.eps[0].real(); });
    add_mode(modes, fns, "reduction_plus_tensor_numerator", "TensorNumerator + ReplayIntegralLibrary, Amplitude::evaluate",
             [sp](int l) { LaneState & s = (*sp)[l]; s.replay_lib.rewind();
                           Result r = evaluate(*s.amp, *s.num, s.replay_lib); g_sink += r.eps[0].real(); });
    add_mode(modes, fns, "masters_only", "recorded AvHOneLoop interface calls replayed through ninja::avh_olo, no reduction",
             [sp](int l) { LaneState & s = (*sp)[l]; g_sink += replay_masters(ninja::avh_olo, s.master_tape, s.lane->mu2).real(); });
    add_mode(modes, fns, "numerator_only", "recorded TensorNumerator calls (inputs) replayed, no reduction",
             [sp](int l) { LaneState & s = (*sp)[l]; g_sink += replay_numerator_inputs(*s.num, s.num_tape).real(); });
    add_mode(modes, fns, "full_madloop_entry",
             "MadLoop's entry point ninja_tensor_evaluate_[smat_]cm_ (new TensorNumerator + Amplitude per call, default AvHOneLoop)",
             [sp, nn, rr](int l) { LaneState & s = (*sp)[l]; Result r = entry_point(*s.lane, nn, rr); g_sink += r.eps[0].real(); });
    for (int g = 0; g < MG_COUNT; ++g) {
      bool any = false;
      for (int l = 0; l < L; ++l)
        any = any || !st[l].master_groups[g].calls.empty();
      if (!any)
        continue;
      add_mode(modes, fns, std::string("masters_") + master_group_name[g],
               std::string("recorded ") + master_group_name[g] + " calls only, replayed through ninja::avh_olo",
               [sp, g](int l) { LaneState & s = (*sp)[l]; g_sink += replay_masters(ninja::avh_olo, s.master_groups[g], s.lane->mu2).real(); });
    }
    // calibration (also warms up), then one discarded warm-up block per mode
    for (size_t m = 0; m < modes.size(); ++m)
      modes[m].iterations = calibrate(fns[m], L, block_ns);
    for (size_t m = 0; m < modes.size(); ++m)
      run_block(fns[m], L, modes[m].iterations);
    for (int r = 0; r < rounds; ++r)
      for (size_t m = 0; m < modes.size(); ++m)
        modes[m].ns_per_point.push_back(run_block(fns[m], L, modes[m].iterations) / (modes[m].iterations * L));
  }

  // Report.
  std::ofstream out(out_path.c_str());
  out << "{\n";
  out << "\"format\": \"ninjars-reference-ninja-bench-v1\",\n";
  out << "\"case\": " << jstr(c.name) << ",\n";
  out << "\"case_file\": " << jstr(case_path) << ",\n";
  out << "\"build\": " << jstr(label) << ",\n";
  out << "\"n\": " << c.n << ", \"rank\": " << c.rank << ", \"lanes_count\": " << L << ",\n";
  out << "\"amplitude\": \"ninja::Amplitude<ninja::ComplexMasses>(n, rank, V, m2); setRenormalizationScale(mu_squared)"
         " (Ninja passes mu_R^2 to IntegralLibrary::init, AvHOneLoop uses mu_R=sqrt); setSMatrix if s_matrix given\",\n";
  out << "\"all_replays_bitwise_identical\": " << (all_identical ? "true" : "false") << ",\n";
  out << "\"lanes\": [\n";
  for (int l = 0; l < L; ++l) {
    LaneState & s = st[l];
    const Result & r = reference[l];
    int ncalls[NK_COUNT] = {0, 0, 0, 0};
    for (size_t i = 0; i < s.num_tape.calls.size(); ++i)
      ncalls[s.num_tape.calls[i].kind]++;
    std::map<std::string, int> mcalls;
    for (size_t i = 0; i < s.master_tape.calls.size(); ++i)
      mcalls[master_kind_name[s.master_tape.calls[i].kind]]++;
    out << " {\"id\": " << jstr(s.lane->id) << ", \"return_status\": " << r.status
        << ",\n  \"eps0\": " << jc(r.eps[0]) << ", \"eps_m1\": " << jc(r.eps[1]) << ", \"eps_m2\": " << jc(r.eps[2])
        << ",\n  \"cut_constructible\": " << jc(r.cc) << ", \"rational\": " << jc(r.rat)
        << ",\n  \"checks\": " << lane_checks[l]
        << ",\n  \"numerator_calls\": {";
    for (int k = 0; k < NK_COUNT; ++k)
      out << (k ? ", " : "") << jstr(num_kind_name[k]) << ": " << ncalls[k];
    out << ", \"total\": " << s.num_tape.calls.size() << ", \"output_values\": " << s.num_tape.outputs.size() << "}";
    out << ",\n  \"master_calls\": {";
    bool first = true;
    for (std::map<std::string, int>::iterator it = mcalls.begin(); it != mcalls.end(); ++it) {
      out << (first ? "" : ", ") << jstr(it->first) << ": " << it->second;
      first = false;
    }
    out << (first ? "" : ", ") << "\"total\": " << s.master_tape.calls.size() << ", \"init_calls\": " << s.master_tape.inits.size() << "}";
    out << ",\n  \"master_call_list\": [";
    size_t pos = 0;
    for (size_t i = 0; i < s.master_tape.calls.size(); ++i) {
      const MasterCall & k = s.master_tape.calls[i];
      out << (i ? ",\n    " : "\n    ") << "{\"kind\": " << jstr(master_kind_name[k.kind]) << ", \"mu_squared\": " << jnum(k.mu2)
          << ", \"s\": [";
      for (int j = 0; j < 6; ++j) out << (j ? "," : "") << jnum(k.s[j]);
      out << "], \"m2\": [";
      for (int j = 0; j < 4; ++j) out << (j ? "," : "") << jc(k.m[j]);
      out << "], \"result\": [";
      for (int j = 0; j < master_nout(k.kind); ++j) out << (j ? "," : "") << jc(s.master_tape.outputs[pos++]);
      out << "]}";
    }
    out << "]}" << (l + 1 < L ? ",\n" : "\n");
  }
  out << "]";
  if (timing) {
    out << ",\n\"timing\": {\n \"unit\": \"ns per phase-space point (one lane evaluation), lanes cycled\",\n";
    out << " \"protocol\": \"one calibration and one discarded warm-up block per mode; then " << rounds
        << " interleaved rounds, each running one block (>= " << block_ms << " ms) of every mode; median over rounds\",\n";
    out << " \"modes\": {\n";
    for (size_t m = 0; m < modes.size(); ++m) {
      out << "  " << jstr(modes[m].name) << ": {\"description\": " << jstr(modes[m].description)
          << ", \"iterations_per_block\": " << modes[m].iterations << ", \"median_ns\": " << jnum(median(modes[m].ns_per_point))
          << ", \"blocks_ns\": [";
      for (size_t b = 0; b < modes[m].ns_per_point.size(); ++b)
        out << (b ? "," : "") << jnum(modes[m].ns_per_point[b]);
      out << "]}" << (m + 1 < modes.size() ? ",\n" : "\n");
    }
    double full = median(modes[0].ns_per_point), pure = median(modes[1].ns_per_point);
    double tred = median(modes[2].ns_per_point), mast = median(modes[3].ns_per_point);
    double numo = median(modes[4].ns_per_point);
    double num_part = tred - pure;
    double sum = pure + num_part + mast;
    out << " },\n \"derived\": {\"full_ns\": " << jnum(full) << ", \"pure_reduction_ns\": " << jnum(pure)
        << ", \"numerator_part_ns\": " << jnum(num_part) << ", \"numerator_only_ns\": " << jnum(numo)
        << ", \"masters_ns\": " << jnum(mast) << ", \"sum_pure_numerator_masters_ns\": " << jnum(sum)
        << ", \"sum_over_full\": " << jnum(sum / full) << "}\n}";
  }
  out << ",\n\"sink\": " << jnum(g_sink) << "\n}\n";
  out.close();
  if (!all_identical) {
    std::cerr << "ERROR: replay results differ from the reference evaluation\n";
    return 1;
  }
  return 0;
}
