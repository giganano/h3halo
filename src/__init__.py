
# from ._fit_driver import fit_driver
from . import _fit_driver

class fit_driver(_fit_driver.fit_driver):

	def __reduce__(self):
		return (self.__class__, (self.sample, self.invcovs, ))
