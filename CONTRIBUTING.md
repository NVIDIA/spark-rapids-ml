# Contributing to Spark Rapids ML
If you are interested in contributing to Spark Rapids ML, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an issue
    describing what you encountered or what you want to see changed.
    - The Spark-Rapids-ML team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.



## Code contributions

### Your first issue
TODO: update after Github repo is created.


### Code Formatting

#### Python
This project follows the
[Black Code Style](https://black.readthedocs.io/en/stable/the_black_code_style/index.html).
The project also requires type annotation.  The tools `isort`, `black`, and `mypy` can be used to check compliance and for auto-formatting.  See [run_test.sh](run_test.sh)

#### Scala

This project follows the official
[Scala style guide](https://docs.scala-lang.org/style/) and the
[Databricks Scala guide](https://github.com/databricks/scala-style-guide), preferring the latter.

#### Java

This project follows the
[Oracle Java code conventions](http://www.oracle.com/technetwork/java/codeconvtoc-136057.html)
and the Scala conventions detailed above, preferring the latter.


### Sign your work

We require that all contributors sign-off on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not signed off will not be accepted.

To sign off on a commit use the `--signoff` (or `-s`) option when committing your changes:

```shell
git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

The sign-off is a simple line at the end of the explanation for the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. Use your real name, no pseudonyms or anonymous contributions.  If you set your `user.name` and `user.email` git configs, you can sign your commit automatically with `git commit -s`.


The signoff means you certify the below (from [developercertificate.org](https://developercertificate.org)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```


### Pull request status checks
A pull request should pass all status checks before merged.
#### signoff check
Please follow the steps in the [Sign your work](#sign-your-work) section,
and make sure at least one commit in your pull request get signed-off.


