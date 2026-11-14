"""Process exit codes shared by every command."""

OK = 0
FAILURE = 1
USAGE = 2
# EX_TEMPFAIL (sysexits.h): the run hit its call budget or a rate limit and saved its state.
# The next scheduled run resumes where this one stopped.
TEMPFAIL = 75
