#include "omill/Tools/LiftRunContract.h"

#include <gtest/gtest.h>

#include <llvm/Support/JSON.h>
#include <llvm/Support/Program.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <llvm/ADT/SmallVector.h>

namespace {

using omill::tools::LiftRunEvent;

bool fileExists(const std::string &p) {
  std::error_code ec;
  return std::filesystem::exists(std::filesystem::path(p), ec);
}

std::vector<LiftRunEvent> parseEvents(const std::string &path) {
  std::ifstream in(path);
  std::vector<LiftRunEvent> events;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto parsed = llvm::json::parse(line);
    if (!parsed)
      continue;
    auto event = omill::tools::parseLiftRunEvent(*parsed);
    if (event)
      events.push_back(*event);
  }
  return events;
}

std::vector<LiftRunEvent> runLiftAndReadEvents(const std::vector<std::string> &args,
                                               const std::string &events_path) {
#if !defined(OMILL_LIFT_EXE_PATH)
  (void) args;
  (void) events_path;
  return {};
#else
  const std::string exe = OMILL_LIFT_EXE_PATH;
  if (!fileExists(exe))
    return {};

  llvm::SmallVector<llvm::StringRef, 32> argv;
  argv.push_back(exe);
  for (const auto &arg : args)
    argv.push_back(arg);
  std::string err;
  bool exec_failed = false;
  (void) llvm::sys::ExecuteAndWait(exe, argv, std::nullopt, {}, 0, 0,
                                   &err, &exec_failed);
  if (!fileExists(events_path))
    return {};
  return parseEvents(events_path);
#endif
}

TEST(OmillLiftEventIntegrationTest, EmitsLifecycleEventsSingleRaw) {
#if !defined(OMILL_LIFT_EXE_PATH)
  GTEST_SKIP() << "OMILL_LIFT_EXE_PATH not defined";
#endif
  const auto tmp = std::filesystem::temp_directory_path();
  const auto stamp = std::to_string(std::time(nullptr));
  const auto raw_path = (tmp / ("omill_lift_raw_" + stamp + ".bin")).string();
  const auto out_path = (tmp / ("omill_lift_raw_" + stamp + ".ll")).string();
  const auto events_path = (tmp / ("omill_lift_raw_" + stamp + ".jsonl")).string();

  {
    std::ofstream raw(raw_path, std::ios::binary);
    const unsigned char bytes[] = {0xC3, 0x90, 0xC3};  // ret; nop; ret
    raw.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
  }

  const std::vector<std::string> args = {
      raw_path, "--raw", "--base", "0x1000", "--va", "0x1000", "--no-abi",
      "-o", out_path, "--event-jsonl", events_path, "--event-detail", "basic"};

  const auto events = runLiftAndReadEvents(args, events_path);
  ASSERT_FALSE(events.empty()) << "No events were emitted";

  bool saw_started = false;
  bool saw_input_loaded = false;
  bool saw_done = false;
  uint64_t prev_seq = 0;
  for (const auto &e : events) {
    if (e.kind == "run_started")
      saw_started = true;
    if (e.kind == "input_load_completed")
      saw_input_loaded = true;
    if (e.kind == "run_done")
      saw_done = true;
    EXPECT_GE(e.seq, prev_seq);
    prev_seq = e.seq;
  }

  EXPECT_TRUE(saw_started);
  EXPECT_TRUE(saw_input_loaded);
  EXPECT_TRUE(saw_done);
}

TEST(OmillLiftEventIntegrationTest, EmitsBatchModeEvents) {
#if !defined(OMILL_LIFT_EXE_PATH)
  GTEST_SKIP() << "OMILL_LIFT_EXE_PATH not defined";
#endif
  const auto tmp = std::filesystem::temp_directory_path();
  const auto stamp = std::to_string(std::time(nullptr) + 1);
  const auto raw_path = (tmp / ("omill_lift_batch_" + stamp + ".bin")).string();
  const auto out_path = (tmp / ("omill_lift_batch_" + stamp + ".ll")).string();
  const auto events_path = (tmp / ("omill_lift_batch_" + stamp + ".jsonl")).string();
  const auto targets_path = (tmp / ("omill_lift_batch_" + stamp + ".json")).string();

  {
    std::ofstream raw(raw_path, std::ios::binary);
    const unsigned char bytes[] = {0xC3, 0xC3};
    raw.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
  }
  {
    std::ofstream t(targets_path, std::ios::binary);
    t << "{\"functions\":[{\"va\":\"0x1000\"}]}";
  }

  const std::vector<std::string> args = {
      raw_path,
      "--raw",
      "--base",
      "0x1000",
      "--no-abi",
      "--deobf-targets",
      targets_path,
      "-o",
      out_path,
      "--event-jsonl",
      events_path,
      "--event-detail",
      "detailed"};

  const auto events = runLiftAndReadEvents(args, events_path);
  ASSERT_FALSE(events.empty()) << "No events were emitted";

  bool saw_batch = false;
  bool saw_done = false;
  for (const auto &e : events) {
    if (e.kind == "batch_targets_ready")
      saw_batch = true;
    if (e.kind == "run_done")
      saw_done = true;
  }

  EXPECT_TRUE(saw_batch);
  EXPECT_TRUE(saw_done);
}

TEST(OmillLiftEventIntegrationTest, EmitsVmModeEventWhenEnabled) {
#if !defined(OMILL_LIFT_EXE_PATH)
  GTEST_SKIP() << "OMILL_LIFT_EXE_PATH not defined";
#endif
  const auto tmp = std::filesystem::temp_directory_path();
  const auto stamp = std::to_string(std::time(nullptr) + 2);
  const auto raw_path = (tmp / ("omill_lift_vm_" + stamp + ".bin")).string();
  const auto out_path = (tmp / ("omill_lift_vm_" + stamp + ".ll")).string();
  const auto events_path = (tmp / ("omill_lift_vm_" + stamp + ".jsonl")).string();

  {
    std::ofstream raw(raw_path, std::ios::binary);
    const unsigned char bytes[] = {0xC3, 0xC3};
    raw.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
  }

  const std::vector<std::string> args = {
      raw_path, "--raw", "--base", "0x1000", "--va", "0x1000",
      "--vm-entry", "0x1000", "--no-abi", "-o", out_path,
      "--event-jsonl", events_path, "--event-detail", "detailed"};

  const auto events = runLiftAndReadEvents(args, events_path);
  ASSERT_FALSE(events.empty()) << "No events were emitted";

  bool saw_vm = false;
  bool saw_done = false;
  for (const auto &e : events) {
    if (e.kind == "vm_mode")
      saw_vm = true;
    if (e.kind == "run_done")
      saw_done = true;
  }
  EXPECT_TRUE(saw_vm);
  EXPECT_TRUE(saw_done);
}

}  // namespace
