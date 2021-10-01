Contributing to zDNN
==========================

License
-------
All contributions have to be submitted under the Apache 2.0 license. See also
the [LICENSE](LICENSE) file.

Developer's Certificate of Origin and Signed-off-by
---------------------------------------------------
The sign-off is a simple line at the end of the explanation for the patch,
which certifies that you wrote it or otherwise have the right to pass it on as
an open-source patch.

With the Signed-off-by line you certify the below:

```
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

If you can certify the above, just add a line stating the following at the
bottom of each of your commit messages:

```
Signed-off-by: Random Developer <random@developer.example.org>
```

Please use your real name and a valid e-mail address (no pseudonyms or anonymous
contributions).

Submitting code
---------------
The preferred way is to create GitHub pull requests for your code contributions.
Please create separate pull requests for each logical enhancement, new feature,
or fix.

GitHub workflow for contributions
---------------------------------
In the examples below we use this fictive identity:

 - Name: Random Developer
 - E-mail: random@developer.example.org
 - GitHub ID: random-developer

### Setup GitHub and local git

1. Create a fork of this repository by clicking the `Fork` button on the top
   right of the [zDNN](https://github.com/IBM/zDNN)
   main page

2. Clone your forked repository to your local development system
   ```
   $ git clone https://github.com/random-developer/zDNN.git
   ```

3. Configure a remote called "upstream" pointing to the official
   zDNN repository on GitHub
   ```
   $ cd zDNN
   ~/zDNN $ git remote add upstream https://github.com/IBM/zDNN.git
   ```

4. Verify your remotes
   ```
   ~/zDNN $ git remote -v
   origin  https://github.com/random-developer/zDNN.git (fetch)
   origin  https://github.com/random-developer/zDNN.git (push)
   upstream        https://github.com/IBM/zDNN.git (fetch)
   upstream        https://github.com/IBM/zDNN.git (push)
   ```
   You now have two remotes: The "origin" remote points to your fork
   and the "upstream" remote to the official zDNN repository.

5. Configure your git user name and e-mail
   ```
   ~/zDNN $ git config user.name "Random Developer"
   ~/zDNN $ git config user.email "random@developer.example.com"
   ```

### Create a pull request

1. Create and checkout a new branch for your contribution
   ```
   ~/zDNN $ git checkout -b contrib-doc-pr
   ```

2. Make your changes to the code
   ```
   ~/zDNN $ vim CONTRIBUTING.md
   ```

3. Build and test your contribution, recommended on NNPA enabled machine.
   ```
   ~/zDNN $ make clean all
   ```

4. Commit your changes
   ```
   ~/zDNN $ git add CONTRIBUTING.md
   ~/zDNN $ git commit -s
   ```

   Provide a meaningful commit message including your "Signed-off-by" line to
   each commit:
   ```
   CONTRIBUTING: Outline steps to submit code

   Explain in more detail how to submit zDNN contributions as GitHub
   pull requests.

   Signed-off-by: Random Developer <random@developer.example.com>
   ```

5. Push the changes to your fork of the repository
   ```
   ~/zDNN $ git push origin contrib-doc-pr
   ```

6. Go to the GitHub website of your zDNN fork and create a pull request
   for your branch "contrib-doc-pr"

### Update a pull request during review

If there are changes requested during the review process, you have to update
your code in the pull request.

To retain the existing review comments, add commits on top of your pull request
branch. Depending on the size and number of changes, a rebase of the pull
request might be required. This will be communicated during the review.

1. Update your code with new commits
   ```
   ~/zDNN $ vi CONTRIBUTING.md
   ~/zDNN $ git add CONTRIBUTING.md
   ~/zDNN $ git commit -s -m "CONTRIBUTING: Add update PR info"
   ```

2. Update your pull request by pushing changes
   ```
   ~/zDNN $ git push origin contrib-doc-pr
   ```

### Finalize a pull request

After the review process is finished or if you are explicitly asked for it,
you have to create a clean commit series.

1. Save branch to "contrib-doc-pr.v1"
   ```
   $ cd zDNN
   ~/zDNN $ git branch contrib-doc-pr.v1
   ```

2. Use interactive git rebase to merge commits, adjust commit messages,
   and rebase onto your local main branch
   ```
   ~/zDNN $ git rebase -i main
   ```

   An editor is started and shows the following:
   ```
   pick 2c73b9fc CONTRIBUTING: Outline steps to submit code
   pick fcfb0412 CONTRIBUTING: Add update PR info
   ```

   To merge the update into the original commit, replace "pick fcfb0412"
   with "squash fcfb0412".

   ```
   pick 2c73b9fc CONTRIBUTING: Outline steps to submit code
   squash fcfb0412 CONTRIBUTING: Add update PR info
   ```

   Save the document and exit the editor to finish the merge. Another editor
   window is presented to modify the commit message.

   You now could change the commit message as follows:

   ```
   CONTRIBUTING: Outline steps to submit code

   Explain in more detail how to submit zDNN contributions as GitHub
   pull requests and how to update already submitted pull requests.

   Signed-off-by: Random Developer <random@developer.example.com>
   ```

   With interactive rebasing you can also change the order of commits and
   modify commit messages with "reword".

3. Use `git push` with the force option to replace the existing pull request
   with your locally modified commits
   ```
   ~/zDNN $ git push --force origin contrib-doc-pr
   ```

### Rebase a pull request

If changes are made to the main branch in the official zDNN
repository you may be asked to rebase your branch with your contribution
onto it. This can be required to prevent any merge conflicts that might
arise when integrating your contribution.

1. Fetch all upstream changes from the official zDNN repository,
   rebase your local main branch and update the main branch
   on your fork
   ```
   ~/zDNN $ git fetch upstream
   ~/zDNN $ git checkout main
   ~/zDNN $ git rebase upstream/main
   ~/zDNN $ git push origin main
   ```

2. Rebase your branch with your contribution onto the main branch of
   the official zDNN repository
   ```
   ~/zDNN $ git checkout contrib-doc-pr
   ~/zDNN $ git rebase main
   ```

3. Use `git push` with the force option to replace the existing pull
   request with your locally modified commits
   ```
   ~/zDNN $ git push --force origin contrib-doc-pr
   ```
