class Localllm < Formula
  include Language::Python::Virtualenv

  desc "One command runs the best local AI your PC can handle, picked from measured scores in your language"
  homepage "https://github.com/phonology024/make-localllm-easier"
  url "https://files.pythonhosted.org/packages/source/m/make-localllm-easier/make_localllm_easier-0.7.1.tar.gz"
  sha256 "fb201722ae8af0a97afabeb486ea520c7dbff5e62fd4788937f93ccc8b6c4ec8"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/localllm --version")
    assert_match "qwen3.8-27b-q3", shell_output("#{bin}/localllm list")
  end
end
