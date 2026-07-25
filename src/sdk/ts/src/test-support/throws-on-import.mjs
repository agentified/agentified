// A module that blows up the moment it is evaluated. `isPackageInstalled` must still
// report it as present: the probe resolves, it never imports. If the probe ever starts
// executing what it resolves, importing this fixture is what will catch it.
throw new Error("throws-on-import.mjs was evaluated; the resolution probe must not import");
